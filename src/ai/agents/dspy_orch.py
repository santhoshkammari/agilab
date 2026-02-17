"""
Multi-Agent Self-Driving Codebase System (DSPy Implementation)
==============================================================

Based on Cursor's research: "Towards Self-Driving Codebases"

Architecture:
  - Recursive tree of Planner agents (plan only, never code)
  - Isolated Worker agents (code only, zero system awareness)
  - Structured Handoff mechanism (reports flow upward through planner tree)
  - Continuous orchestration loop (runs for hours autonomously)

Design Principles:
  1. Anti-fragile: individual agent failures don't break the system
  2. Throughput-first: accept small error rate, avoid gridlock from perfection demands
  3. Empirical: every design choice driven by observed agent behavior

Usage:
    import dspy
    from multi_agent_system import MultiAgentOrchestrator

    dspy.configure(lm=dspy.LM("openai/gpt-4o", max_tokens=4096))

    orchestrator = MultiAgentOrchestrator(
        repo_path="/path/to/project",
        user_instructions="Build a REST API with user auth, CRUD for posts, and rate limiting",
        max_workers=20,
        max_depth=3,
    )
    orchestrator.run()
"""

import dspy
import uuid
import os
import json
import shutil
import subprocess
import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_agent.log"),
    ],
)
logger = logging.getLogger("multi_agent")


# ===========================================================================
# 1. DATA STRUCTURES
# ===========================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    CRITICAL = 1      # Blocks many other tasks
    HIGH = 2          # Important for progress
    NORMAL = 3        # Standard work
    LOW = 4           # Nice-to-have, polish


@dataclass
class Task:
    """A unit of work emitted by a planner and executed by a worker."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    scope: str = ""                          # What files/modules this touches
    acceptance_criteria: str = ""             # How to know the task is done
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    planner_id: str = ""                     # Which planner created this task
    worker_id: Optional[str] = None          # Which worker picked it up
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class Handoff:
    """
    Structured report from a worker back to its parent planner.
    This is THE key communication mechanism — information flows UP only.
    """
    task_id: str = ""
    task_description: str = ""
    what_was_done: str = ""                  # Summary of changes made
    files_modified: list = field(default_factory=list)
    code_diff_summary: str = ""              # High-level diff description
    test_results: str = ""                   # Did tests pass? Which ones?
    concerns: str = ""                       # Potential issues the worker noticed
    deviations: str = ""                     # How the result differs from the task spec
    discoveries: str = ""                    # Unexpected findings about the codebase
    suggestions: str = ""                    # Ideas for follow-up work
    success: bool = True
    error_info: str = ""                     # If failed, what went wrong


@dataclass
class PlannerState:
    """Tracks the state of a planner agent in the tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    scope: str = ""                          # What this planner owns
    goal: str = ""                           # The specific goal for this planner
    parent_id: Optional[str] = None          # None for root planner
    depth: int = 0                           # 0 = root
    scratchpad: str = ""                     # Frequently REWRITTEN (not appended)
    child_planner_ids: list = field(default_factory=list)
    pending_handoffs: list = field(default_factory=list)  # Handoffs waiting to be processed
    tasks_emitted: int = 0
    tasks_completed: int = 0
    is_complete: bool = False
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    iteration_count: int = 0


@dataclass
class SystemMetrics:
    """Observability into the running system."""
    total_commits: int = 0
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tool_calls: int = 0
    active_workers: int = 0
    active_planners: int = 0
    start_time: float = field(default_factory=time.time)
    errors: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    @property
    def commits_per_hour(self) -> float:
        hours = self.uptime_hours
        return self.total_commits / hours if hours > 0 else 0

    @property
    def error_rate(self) -> float:
        total = self.total_tasks_completed + self.total_tasks_failed
        return self.total_tasks_failed / total if total > 0 else 0

    def record_commit(self):
        with self._lock:
            self.total_commits += 1

    def record_task_created(self):
        with self._lock:
            self.total_tasks_created += 1

    def record_task_completed(self):
        with self._lock:
            self.total_tasks_completed += 1

    def record_task_failed(self):
        with self._lock:
            self.total_tasks_failed += 1

    def log_summary(self):
        logger.info(
            f"[METRICS] uptime={self.uptime_hours:.1f}h | "
            f"commits={self.total_commits} ({self.commits_per_hour:.0f}/hr) | "
            f"tasks={self.total_tasks_completed}/{self.total_tasks_created} done | "
            f"error_rate={self.error_rate:.1%} | "
            f"workers={self.active_workers} planners={self.active_planners}"
        )


# ===========================================================================
# 2. REPOSITORY MANAGEMENT
# ===========================================================================

class RepoManager:
    """
    Manages git operations for the multi-agent system.

    Key design: each worker gets its OWN copy of the repo (git worktree).
    No shared mutable state. No locks between workers.
    Workers commit independently — some merge turbulence is accepted.
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.worktrees_dir = self.repo_path / ".agent_worktrees"
        self.worktrees_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()  # Only for worktree creation/cleanup
        self._ensure_git_repo()

    def _ensure_git_repo(self):
        """Initialize git repo if not already one."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            self._run_git(["init"], cwd=self.repo_path)
            self._run_git(["checkout", "-b", "main"], cwd=self.repo_path)
            # Initial commit so worktrees work
            readme = self.repo_path / "README.md"
            if not readme.exists():
                readme.write_text("# Project\n")
            self._run_git(["add", "."], cwd=self.repo_path)
            self._run_git(["commit", "-m", "Initial commit"], cwd=self.repo_path)

    def _run_git(self, args: list, cwd: Path = None) -> str:
        """Run a git command and return stdout."""
        cwd = cwd or self.repo_path
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0 and "already exists" not in result.stderr:
                logger.warning(f"Git command failed: git {' '.join(args)}\n{result.stderr}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: git {' '.join(args)}")
            return ""

    def create_worker_copy(self, worker_id: str) -> Path:
        """
        Create an isolated copy of the repo for a worker.
        Uses git worktree for efficiency (shared .git objects).
        Falls back to full copy if worktree fails.
        """
        worker_dir = self.worktrees_dir / f"worker_{worker_id}"
        branch_name = f"worker/{worker_id}"

        with self._lock:
            if worker_dir.exists():
                shutil.rmtree(str(worker_dir), ignore_errors=True)

            # Create branch + worktree
            self._run_git(["branch", "-D", branch_name], cwd=self.repo_path)
            self._run_git(
                ["worktree", "add", "-b", branch_name, str(worker_dir), "main"],
                cwd=self.repo_path,
            )

        if not worker_dir.exists():
            # Fallback: simple copy
            logger.warning(f"Worktree failed for {worker_id}, falling back to copy")
            shutil.copytree(str(self.repo_path), str(worker_dir), ignore=shutil.ignore_patterns('.agent_worktrees'))

        return worker_dir

    def commit_worker_changes(self, worker_id: str, message: str) -> bool:
        """
        Commit worker changes and merge to main.
        Accept merge turbulence — other agents will fix conflicts.
        """
        worker_dir = self.worktrees_dir / f"worker_{worker_id}"
        if not worker_dir.exists():
            return False

        # Stage and commit in worker branch
        self._run_git(["add", "-A"], cwd=worker_dir)
        diff = self._run_git(["diff", "--cached", "--stat"], cwd=worker_dir)
        if not diff:
            logger.info(f"Worker {worker_id}: no changes to commit")
            return False

        self._run_git(["commit", "-m", f"[agent/{worker_id}] {message}"], cwd=worker_dir)

        # Merge to main (accept some failures — anti-fragile)
        with self._lock:
            branch_name = f"worker/{worker_id}"
            result = self._run_git(["merge", branch_name, "--no-edit"], cwd=self.repo_path)
            if "CONFLICT" in result or "conflict" in result.lower():
                # Abort conflicting merge — another agent will handle later
                self._run_git(["merge", "--abort"], cwd=self.repo_path)
                logger.warning(f"Worker {worker_id}: merge conflict, skipping (anti-fragile)")
                return False

        return True

    def cleanup_worker_copy(self, worker_id: str):
        """Remove a worker's worktree and branch."""
        worker_dir = self.worktrees_dir / f"worker_{worker_id}"
        branch_name = f"worker/{worker_id}"
        with self._lock:
            if worker_dir.exists():
                self._run_git(["worktree", "remove", str(worker_dir), "--force"],
                              cwd=self.repo_path)
            self._run_git(["branch", "-D", branch_name], cwd=self.repo_path)

    def get_codebase_snapshot(self) -> str:
        """
        Get a summary of the current codebase state for planners.
        Includes: file tree, recent commits, and any test results.
        """
        # File tree (max depth 3)
        file_tree = self._run_git(
            ["ls-tree", "-r", "--name-only", "HEAD"], cwd=self.repo_path
        )

        # Recent commits
        recent_commits = self._run_git(
            ["log", "--oneline", "-20"], cwd=self.repo_path
        )

        # Current branch status
        status = self._run_git(["status", "--short"], cwd=self.repo_path)

        snapshot = f"""## Current Codebase State

### File Tree
{file_tree[:3000] if file_tree else "(empty repo)"}

### Recent Commits (last 20)
{recent_commits if recent_commits else "(no commits yet)"}

### Working Directory Status
{status if status else "(clean)"}
"""
        return snapshot

    def get_file_contents(self, filepath: str, cwd: Path = None) -> str:
        """Read a file from the repo."""
        target = (cwd or self.repo_path) / filepath
        try:
            return target.read_text(errors="replace")[:10000]  # Cap per file
        except (FileNotFoundError, IsADirectoryError):
            return f"(file not found: {filepath})"


# ===========================================================================
# 3. DSPy SIGNATURES — The LLM-powered decision points
# ===========================================================================

class AnalyzeCodebase(dspy.Signature):
    """Analyze the current state of a codebase to identify what exists,
    what's missing, what's broken, and what should be prioritized next.
    Be specific and actionable. Focus on gaps between current state and the goal."""

    goal: str = dspy.InputField(desc="The overall goal or user instructions")
    codebase_state: str = dspy.InputField(desc="Current file tree, recent commits, status")
    recent_handoffs: str = dspy.InputField(desc="Recent worker reports summarizing completed work")
    analysis: str = dspy.OutputField(desc="Detailed analysis: what exists, what's missing, what's broken")
    priorities: str = dspy.OutputField(desc="Ordered list of the most impactful things to do next")


class PlanTasks(dspy.Signature):
    """Given an analysis of the codebase, generate specific, actionable tasks
    that workers can execute independently and in parallel.

    CONSTRAINTS:
    - Each task must be completable by ONE worker in isolation
    - Tasks should be specific: name exact files, functions, modules
    - Tasks must NOT depend on each other within this batch
    - Generate 5-50 tasks depending on scope (be ambitious, not conservative)
    - No vague tasks like "improve code" — be precise about what and how"""

    goal: str = dspy.InputField(desc="The overall goal for this planner's scope")
    analysis: str = dspy.InputField(desc="Current codebase analysis and priorities")
    scratchpad: str = dspy.InputField(desc="Planner's current scratchpad with context")
    tasks_json: str = dspy.OutputField(
        desc='JSON array of tasks. Each: {"description": "...", "scope": "files/modules touched", '
             '"acceptance_criteria": "...", "priority": "critical|high|normal|low"}'
    )
    subplanner_scopes_json: str = dspy.OutputField(
        desc='JSON array of scopes that need their own subplanner (large independent subsystems). '
             'Each: {"scope": "...", "goal": "..."}. Empty array [] if no subdivision needed.'
    )
    updated_scratchpad: str = dspy.OutputField(
        desc="REWRITTEN scratchpad (not appended). Capture current understanding, decisions, "
             "and what to focus on next. Max 500 words."
    )


class ExecuteTask(dspy.Signature):
    """You are a worker agent. Execute the assigned coding task completely.

    CONSTRAINTS:
    - You have NO awareness of other workers or the larger system
    - Work ONLY within the scope described in the task
    - Do NOT fix unrelated issues you notice (report them in discoveries)
    - Implement FULLY — no TODOs, no partial implementations, no placeholders
    - Write or update tests if applicable
    - If the task is impossible or blocked, explain why in your handoff"""

    task_description: str = dspy.InputField(desc="What to implement/fix/change")
    task_scope: str = dspy.InputField(desc="Which files/modules this task touches")
    acceptance_criteria: str = dspy.InputField(desc="How to verify the task is done")
    relevant_code: str = dspy.InputField(desc="Current contents of relevant files")
    file_tree: str = dspy.InputField(desc="Full file tree of the project")

    code_changes: str = dspy.OutputField(
        desc="Complete code to write. Format: one or more blocks of "
             "FILE: path/to/file.ext\n```\n<full file contents>\n```"
    )
    test_results: str = dspy.OutputField(
        desc="Description of what was tested and results"
    )
    handoff_report: str = dspy.OutputField(
        desc="Structured report: (1) What was done (2) Files modified "
             "(3) Concerns/risks (4) Deviations from task spec "
             "(5) Discoveries about the codebase (6) Suggestions for follow-up"
    )


class RewriteScratchpad(dspy.Signature):
    """Rewrite a planner's scratchpad from scratch based on the latest information.
    Do NOT append to the old scratchpad — write a completely fresh one.
    Capture: current understanding, key decisions made, active concerns,
    and what to focus on next. Be concise (max 500 words)."""

    old_scratchpad: str = dspy.InputField(desc="Previous scratchpad content")
    new_information: str = dspy.InputField(desc="Recent handoffs, state changes, and observations")
    goal: str = dspy.InputField(desc="This planner's goal")
    fresh_scratchpad: str = dspy.OutputField(desc="Completely rewritten scratchpad, max 500 words")


class SummarizeContext(dspy.Signature):
    """Summarize a long context into a concise version that preserves
    all critical information: decisions, architecture, known issues,
    and current priorities. Lose nothing important."""

    long_context: str = dspy.InputField(desc="Long context that needs compression")
    goal: str = dspy.InputField(desc="The relevant goal for filtering importance")
    summary: str = dspy.OutputField(desc="Concise summary preserving all critical information")


class ReviewHandoffs(dspy.Signature):
    """Review completed worker handoffs and determine next actions.
    Identify: what was accomplished, what failed, what new issues arose,
    and what follow-up tasks are needed."""

    goal: str = dspy.InputField(desc="This planner's goal")
    handoffs: str = dspy.InputField(desc="Batch of worker handoff reports")
    codebase_state: str = dspy.InputField(desc="Current codebase state after the work")
    assessment: str = dspy.OutputField(desc="Assessment of progress: what succeeded, what needs rework")
    follow_up_needed: bool = dspy.OutputField(desc="Whether more tasks are needed to reach the goal")
    is_scope_complete: bool = dspy.OutputField(desc="Whether this planner's entire scope is done")


# ===========================================================================
# 4. DSPy MODULES — The agent roles
# ===========================================================================

class PlannerAgent(dspy.Module):
    """
    A planner agent that owns a scope and emits tasks.
    Does NO coding itself. Can spawn subplanners for large sub-scopes.

    This is used for both the root planner and recursive subplanners.
    The only difference is the scope/goal and depth.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeCodebase)
        self.plan = dspy.ChainOfThought(PlanTasks)
        self.review = dspy.ChainOfThought(ReviewHandoffs)
        self.rewrite_scratchpad = dspy.ChainOfThought(RewriteScratchpad)
        self.summarize = dspy.ChainOfThought(SummarizeContext)

    def analyze_and_plan(
        self,
        state: PlannerState,
        codebase_snapshot: str,
        handoff_texts: str = "",
    ) -> tuple[list[Task], list[dict], str]:
        """
        Core planner cycle:
        1. Analyze current codebase state
        2. Generate tasks + identify sub-scopes needing subplanners
        3. Update scratchpad (rewrite, not append)

        Returns: (tasks, subplanner_scopes, updated_scratchpad)
        """
        state.iteration_count += 1
        state.last_active = time.time()

        # Step 1: Analyze
        analysis_result = self.analyze(
            goal=state.goal,
            codebase_state=codebase_snapshot,
            recent_handoffs=handoff_texts or "(no recent handoffs)",
        )

        # Step 2: Plan tasks + identify subdivision opportunities
        plan_result = self.plan(
            goal=state.goal,
            analysis=f"{analysis_result.analysis}\n\nPriorities:\n{analysis_result.priorities}",
            scratchpad=state.scratchpad or "(fresh start)",
        )

        # Parse tasks JSON
        tasks = self._parse_tasks(plan_result.tasks_json, state.id)

        # Parse subplanner scopes
        subplanner_scopes = self._parse_json_safe(
            plan_result.subplanner_scopes_json, []
        )

        # Step 3: Update scratchpad (REWRITE)
        state.scratchpad = plan_result.updated_scratchpad
        state.tasks_emitted += len(tasks)

        logger.info(
            f"[PLANNER {state.id}] depth={state.depth} iter={state.iteration_count} | "
            f"emitted {len(tasks)} tasks, {len(subplanner_scopes)} subplanner scopes"
        )

        return tasks, subplanner_scopes, state.scratchpad

    def process_handoffs(
        self,
        state: PlannerState,
        handoffs: list[Handoff],
        codebase_snapshot: str,
    ) -> tuple[bool, bool]:
        """
        Process completed worker handoffs.
        Returns: (follow_up_needed, is_scope_complete)
        """
        if not handoffs:
            return True, False

        # Format handoffs for the LLM
        handoff_text = "\n\n---\n\n".join(
            f"### Task: {h.task_description}\n"
            f"**Done:** {h.what_was_done}\n"
            f"**Files:** {', '.join(h.files_modified)}\n"
            f"**Tests:** {h.test_results}\n"
            f"**Concerns:** {h.concerns}\n"
            f"**Deviations:** {h.deviations}\n"
            f"**Discoveries:** {h.discoveries}\n"
            f"**Suggestions:** {h.suggestions}\n"
            f"**Success:** {h.success}"
            for h in handoffs
        )

        result = self.review(
            goal=state.goal,
            handoffs=handoff_text,
            codebase_state=codebase_snapshot,
        )

        state.tasks_completed += len([h for h in handoffs if h.success])

        logger.info(
            f"[PLANNER {state.id}] reviewed {len(handoffs)} handoffs | "
            f"follow_up={result.follow_up_needed} complete={result.is_scope_complete}"
        )

        return result.follow_up_needed, result.is_scope_complete

    def refresh_scratchpad(self, state: PlannerState, new_info: str):
        """Rewrite scratchpad with fresh context (freshness mechanism)."""
        result = self.rewrite_scratchpad(
            old_scratchpad=state.scratchpad,
            new_information=new_info,
            goal=state.goal,
        )
        state.scratchpad = result.fresh_scratchpad

    def compress_context(self, state: PlannerState) -> str:
        """Summarize when approaching context limits."""
        result = self.summarize(
            long_context=state.scratchpad,
            goal=state.goal,
        )
        state.scratchpad = result.summary
        return result.summary

    def _parse_tasks(self, tasks_json: str, planner_id: str) -> list[Task]:
        """Parse task JSON from LLM output into Task objects."""
        raw = self._parse_json_safe(tasks_json, [])
        tasks = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "normal": TaskPriority.NORMAL,
                "low": TaskPriority.LOW,
            }
            tasks.append(Task(
                description=item.get("description", ""),
                scope=item.get("scope", ""),
                acceptance_criteria=item.get("acceptance_criteria", ""),
                priority=priority_map.get(item.get("priority", "normal"), TaskPriority.NORMAL),
                planner_id=planner_id,
            ))
        return tasks

    @staticmethod
    def _parse_json_safe(text: str, default):
        """Safely parse JSON from LLM output, handling common quirks."""
        if not text:
            return default
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON: {cleaned[:200]}...")
            return default


class WorkerAgent(dspy.Module):
    """
    An isolated worker agent that executes a single task.

    Key properties:
    - Works on its OWN copy of the repo
    - Has ZERO awareness of other workers or the system
    - Produces a structured Handoff on completion
    - Does NOT fix unrelated issues (reports them in discoveries)
    """

    def __init__(self):
        super().__init__()
        self.execute = dspy.ChainOfThought(ExecuteTask)

    def run_task(
        self,
        task: Task,
        worker_dir: Path,
        repo_manager: RepoManager,
    ) -> Handoff:
        """
        Execute a task in an isolated repo copy.
        Returns a structured Handoff regardless of success or failure.
        """
        worker_id = str(uuid.uuid4())[:8]
        task.worker_id = worker_id
        task.status = TaskStatus.IN_PROGRESS

        logger.info(f"[WORKER {worker_id}] starting: {task.description[:80]}...")

        try:
            # Read relevant files for context
            file_tree = self._get_file_tree(worker_dir)
            relevant_code = self._gather_relevant_code(worker_dir, task.scope, file_tree)

            # Execute the task via LLM
            result = self.execute(
                task_description=task.description,
                task_scope=task.scope,
                acceptance_criteria=task.acceptance_criteria,
                relevant_code=relevant_code,
                file_tree=file_tree,
            )

            # Apply code changes to the worker's repo copy
            files_modified = self._apply_code_changes(worker_dir, result.code_changes)

            # Commit and merge to main
            committed = repo_manager.commit_worker_changes(
                worker_id=f"{task.id}_{worker_id}",
                message=task.description[:72],
            )

            task.status = TaskStatus.COMPLETED

            # Parse handoff report into structured fields
            handoff = self._build_handoff(
                task=task,
                handoff_text=result.handoff_report,
                files_modified=files_modified,
                test_results=result.test_results,
                committed=committed,
            )

            logger.info(f"[WORKER {worker_id}] completed: {task.description[:50]}... "
                        f"({len(files_modified)} files)")
            return handoff

        except Exception as e:
            logger.error(f"[WORKER {worker_id}] FAILED: {e}")
            task.status = TaskStatus.FAILED
            return Handoff(
                task_id=task.id,
                task_description=task.description,
                what_was_done="Task failed with an error",
                success=False,
                error_info=str(e),
            )

    def _get_file_tree(self, worker_dir: Path) -> str:
        """Get file tree of the worker's repo copy."""
        tree_lines = []
        for root, dirs, files in os.walk(worker_dir):
            # Skip hidden dirs and common noise
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       ("node_modules", "target", "__pycache__", "venv", ".agent_worktrees")]
            rel_root = os.path.relpath(root, worker_dir)
            depth = rel_root.count(os.sep)
            if depth > 4:
                continue
            indent = "  " * depth
            if rel_root != ".":
                tree_lines.append(f"{indent}{os.path.basename(root)}/")
            for f in files[:50]:  # Cap files per dir
                tree_lines.append(f"{indent}  {f}")
        return "\n".join(tree_lines[:200])  # Cap total lines

    def _gather_relevant_code(self, worker_dir: Path, scope: str, file_tree: str) -> str:
        """Read files mentioned in the task scope."""
        relevant = []
        # Extract file paths from scope description
        for token in scope.replace(",", " ").replace(";", " ").split():
            token = token.strip("\"'`()")
            if "/" in token or "." in token:
                filepath = worker_dir / token
                if filepath.is_file():
                    content = filepath.read_text(errors="replace")[:5000]
                    relevant.append(f"### FILE: {token}\n```\n{content}\n```")

        # If no specific files found, try to find relevant ones from file tree
        if not relevant:
            # Read up to 5 files that seem related to the scope keywords
            keywords = [w.lower() for w in scope.split() if len(w) > 3]
            for line in file_tree.split("\n"):
                fname = line.strip().rstrip("/")
                if any(kw in fname.lower() for kw in keywords):
                    filepath = self._find_file(worker_dir, fname)
                    if filepath and filepath.is_file():
                        content = filepath.read_text(errors="replace")[:5000]
                        relevant.append(f"### FILE: {fname}\n```\n{content}\n```")
                    if len(relevant) >= 5:
                        break

        return "\n\n".join(relevant) if relevant else "(no existing files found for this scope)"

    def _find_file(self, base: Path, name: str) -> Optional[Path]:
        """Find a file by name in the directory tree."""
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if name in files:
                return Path(root) / name
        return None

    def _apply_code_changes(self, worker_dir: Path, code_changes: str) -> list[str]:
        """Parse and write code changes to files."""
        files_modified = []
        current_file = None
        current_content = []
        in_code_block = False

        for line in code_changes.split("\n"):
            # Detect FILE: markers
            if line.strip().startswith("FILE:"):
                # Save previous file
                if current_file and current_content:
                    self._write_file(worker_dir, current_file, current_content)
                    files_modified.append(current_file)
                current_file = line.strip().replace("FILE:", "").strip()
                current_content = []
                in_code_block = False
            elif line.strip().startswith("```") and current_file:
                if in_code_block:
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block and current_file:
                current_content.append(line)

        # Save last file
        if current_file and current_content:
            self._write_file(worker_dir, current_file, current_content)
            files_modified.append(current_file)

        return files_modified

    def _write_file(self, base: Path, filepath: str, lines: list[str]):
        """Write content to a file, creating directories as needed."""
        target = base / filepath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines))

    def _build_handoff(
        self,
        task: Task,
        handoff_text: str,
        files_modified: list[str],
        test_results: str,
        committed: bool,
    ) -> Handoff:
        """Build a structured Handoff from worker output."""
        return Handoff(
            task_id=task.id,
            task_description=task.description,
            what_was_done=handoff_text,
            files_modified=files_modified,
            code_diff_summary=f"Modified {len(files_modified)} files. Committed: {committed}",
            test_results=test_results,
            concerns="",
            deviations="",
            discoveries="",
            suggestions="",
            success=True,
        )


class FreshnessManager(dspy.Module):
    """
    Prevents context drift over long-running sessions.

    Two mechanisms:
    1. Scratchpad rewriting: periodically rewrite (not append) planners' scratchpads
    2. Context summarization: compress when approaching context limits
    """

    def __init__(self, rewrite_interval_iterations: int = 5):
        super().__init__()
        self.rewrite_scratchpad = dspy.ChainOfThought(RewriteScratchpad)
        self.summarize = dspy.ChainOfThought(SummarizeContext)
        self.rewrite_interval = rewrite_interval_iterations

    def should_refresh(self, state: PlannerState) -> bool:
        """Check if a planner needs a scratchpad refresh."""
        return state.iteration_count % self.rewrite_interval == 0 and state.iteration_count > 0

    def should_summarize(self, state: PlannerState) -> bool:
        """Check if a planner's context is getting too large."""
        return len(state.scratchpad) > 3000  # Rough proxy for context pressure

    def refresh(self, state: PlannerState, recent_info: str):
        """Rewrite the scratchpad from scratch."""
        result = self.rewrite_scratchpad(
            old_scratchpad=state.scratchpad,
            new_information=recent_info,
            goal=state.goal,
        )
        state.scratchpad = result.fresh_scratchpad
        logger.info(f"[FRESHNESS] Rewrote scratchpad for planner {state.id}")

    def compress(self, state: PlannerState):
        """Summarize the scratchpad to reduce context size."""
        result = self.summarize(
            long_context=state.scratchpad,
            goal=state.goal,
        )
        old_len = len(state.scratchpad)
        state.scratchpad = result.summary
        logger.info(
            f"[FRESHNESS] Compressed planner {state.id} scratchpad: "
            f"{old_len} → {len(state.scratchpad)} chars"
        )


# ===========================================================================
# 5. THE ORCHESTRATOR — The full system harness
# ===========================================================================

class MultiAgentOrchestrator(dspy.Module):
    """
    The main harness that orchestrates the entire multi-agent system.

    Manages:
    - A recursive tree of PlannerAgents
    - A pool of WorkerAgents executing tasks in parallel
    - Handoff routing from workers back to parent planners
    - Freshness checks to prevent drift
    - Metrics and observability
    - Graceful error recovery (anti-fragile)

    Usage:
        orchestrator = MultiAgentOrchestrator(
            repo_path="/path/to/project",
            user_instructions="Build a REST API with auth and CRUD",
            max_workers=20,
        )
        orchestrator.run()
    """

    def __init__(
        self,
        max_workers: int = 10,
        max_depth: int = 3,
        max_iterations: int = 100,
        freshness_interval: int = 5,
        cycle_delay_seconds: float = 2.0,
    ):
        super().__init__()

        # Configuration
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.cycle_delay = cycle_delay_seconds
        self.freshness_interval = freshness_interval

        # Core components (initialized in run)
        self.planner_agent = PlannerAgent()
        self.worker_agent = WorkerAgent()

    def forward(self, repo_path: str, user_instructions: str):
        """DSPy forward pass — runs the full orchestration."""
        return self.run(repo_path, user_instructions)

    def run(self, repo_path: str, user_instructions: str) -> dict:
        """
        Main entry point. Runs the continuous orchestration loop.
        Returns final metrics when complete.
        """
        self.repo_path = repo_path
        self.user_instructions = user_instructions
        self.repo = RepoManager(repo_path)
        self.freshness = FreshnessManager(rewrite_interval_iterations=self.freshness_interval)
        self.metrics = SystemMetrics()
        self.planners: dict[str, PlannerState] = {}
        self.task_queue = queue.PriorityQueue()
        self.handoff_inbox: dict[str, list[Handoff]] = {}
        self._inbox_lock = threading.Lock()
        self._running = True
        logger.info("=" * 60)
        logger.info(f"MULTI-AGENT SYSTEM STARTING")
        logger.info(f"Repo: {repo_path}")
        logger.info(f"Instructions: {user_instructions[:200]}")
        logger.info(f"Max workers: {self.max_workers} | Max depth: {self.max_depth}")
        logger.info("=" * 60)

        # Initialize root planner
        root = PlannerState(
            scope="entire project",
            goal=user_instructions,
            depth=0,
        )
        self.planners[root.id] = root
        self.handoff_inbox[root.id] = []

        iteration = 0

        try:
            while self._running and iteration < self.max_iterations:
                iteration += 1
                logger.info(f"\n{'='*40} CYCLE {iteration} {'='*40}")

                # --------------------------------------------------------
                # PHASE 1: All planners analyze and emit tasks
                # --------------------------------------------------------
                self._run_planner_phase()

                # --------------------------------------------------------
                # PHASE 2: Workers execute tasks in parallel
                # --------------------------------------------------------
                self._run_worker_phase()

                # --------------------------------------------------------
                # PHASE 3: Route handoffs to parent planners
                # --------------------------------------------------------
                # (Handled automatically in _run_worker_phase)

                # --------------------------------------------------------
                # PHASE 4: Freshness checks on all planners
                # --------------------------------------------------------
                self._run_freshness_phase()

                # --------------------------------------------------------
                # PHASE 5: Check completion + metrics
                # --------------------------------------------------------
                self.metrics.active_planners = len([
                    p for p in self.planners.values() if not p.is_complete
                ])
                self.metrics.log_summary()

                # Check if root planner is done
                if root.is_complete:
                    logger.info("ROOT PLANNER REPORTS: SCOPE COMPLETE")
                    break

                # Check if system is stuck (no tasks, no active planners)
                if (self.task_queue.empty() and
                    all(p.is_complete for p in self.planners.values())):
                    logger.info("All planners complete and no tasks remaining. Done.")
                    break

                time.sleep(self.cycle_delay)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down gracefully...")
        finally:
            self._running = False
            self._cleanup()

        logger.info("=" * 60)
        logger.info("SYSTEM STOPPED")
        self.metrics.log_summary()
        logger.info("=" * 60)

        return {
            "total_commits": self.metrics.total_commits,
            "total_tasks_created": self.metrics.total_tasks_created,
            "total_tasks_completed": self.metrics.total_tasks_completed,
            "error_rate": self.metrics.error_rate,
            "uptime_hours": self.metrics.uptime_hours,
            "iterations": iteration,
        }

    # ------------------------------------------------------------------
    # PHASE 1: Planner Phase
    # ------------------------------------------------------------------

    def _run_planner_phase(self):
        """All active planners analyze codebase and emit tasks."""
        codebase_snapshot = self.repo.get_codebase_snapshot()
        active_planners = [
            p for p in self.planners.values() if not p.is_complete
        ]

        for state in active_planners:
            try:
                # Collect pending handoffs for this planner
                pending = self._drain_handoffs(state.id)

                # If there are handoffs, process them first
                if pending:
                    follow_up, is_complete = self.planner_agent.process_handoffs(
                        state=state,
                        handoffs=pending,
                        codebase_snapshot=codebase_snapshot,
                    )
                    state.is_complete = is_complete
                    if is_complete:
                        logger.info(f"[PLANNER {state.id}] scope complete!")
                        continue
                    if not follow_up:
                        continue  # No more work needed this cycle

                # Analyze and plan
                handoff_text = "\n".join(
                    f"- {h.task_description}: {h.what_was_done}" for h in pending
                ) if pending else ""

                tasks, subplanner_scopes, _ = self.planner_agent.analyze_and_plan(
                    state=state,
                    codebase_snapshot=codebase_snapshot,
                    handoff_texts=handoff_text,
                )

                # Enqueue tasks (sorted by priority)
                for task in tasks:
                    self.task_queue.put((task.priority.value, time.time(), task))
                    self.metrics.record_task_created()

                # Spawn subplanners (recursive, respecting max depth)
                if state.depth < self.max_depth:
                    for sp_spec in subplanner_scopes:
                        if isinstance(sp_spec, dict):
                            self._spawn_subplanner(
                                parent=state,
                                scope=sp_spec.get("scope", ""),
                                goal=sp_spec.get("goal", ""),
                            )

            except Exception as e:
                logger.error(f"[PLANNER {state.id}] error in planning phase: {e}")
                # Anti-fragile: planner failure doesn't crash the system
                continue

    def _spawn_subplanner(self, parent: PlannerState, scope: str, goal: str):
        """Create a recursive subplanner that owns a narrower scope."""
        sub = PlannerState(
            scope=scope,
            goal=goal,
            parent_id=parent.id,
            depth=parent.depth + 1,
        )
        self.planners[sub.id] = sub
        self.handoff_inbox[sub.id] = []
        parent.child_planner_ids.append(sub.id)
        logger.info(
            f"[PLANNER {parent.id}] spawned subplanner {sub.id} "
            f"(depth={sub.depth}): {scope[:60]}"
        )

    # ------------------------------------------------------------------
    # PHASE 2: Worker Phase
    # ------------------------------------------------------------------

    def _run_worker_phase(self):
        """Workers pick up tasks and execute them in parallel."""
        # Gather tasks up to max_workers
        tasks_to_run = []
        while not self.task_queue.empty() and len(tasks_to_run) < self.max_workers:
            try:
                _, _, task = self.task_queue.get_nowait()
                tasks_to_run.append(task)
            except queue.Empty:
                break

        if not tasks_to_run:
            logger.info("[WORKERS] No tasks in queue this cycle")
            return

        logger.info(f"[WORKERS] Executing {len(tasks_to_run)} tasks in parallel")
        self.metrics.active_workers = len(tasks_to_run)

        # Execute in parallel using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_task = {}
            for task in tasks_to_run:
                # Create isolated repo copy for this worker
                worker_id = f"{task.id}_{str(uuid.uuid4())[:4]}"
                worker_dir = self.repo.create_worker_copy(worker_id)

                future = pool.submit(
                    self._execute_worker_task,
                    task=task,
                    worker_dir=worker_dir,
                    worker_id=worker_id,
                )
                future_to_task[future] = (task, worker_id)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task, worker_id = future_to_task[future]
                try:
                    handoff = future.result(timeout=300)  # 5 min timeout

                    # Route handoff to parent planner
                    self._deliver_handoff(task.planner_id, handoff)

                    if handoff.success:
                        self.metrics.record_task_completed()
                        self.metrics.record_commit()
                    else:
                        self.metrics.record_task_failed()
                        # Retry logic (anti-fragile)
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.PENDING
                            self.task_queue.put(
                                (task.priority.value, time.time(), task)
                            )
                            logger.info(
                                f"[RETRY] Task {task.id} retry {task.retry_count}/{task.max_retries}"
                            )

                except Exception as e:
                    logger.error(f"[WORKER] Task {task.id} exception: {e}")
                    self.metrics.record_task_failed()
                    # Deliver failure handoff so planner knows
                    self._deliver_handoff(task.planner_id, Handoff(
                        task_id=task.id,
                        task_description=task.description,
                        what_was_done="Worker crashed",
                        success=False,
                        error_info=str(e),
                    ))
                finally:
                    # Clean up worker copy
                    try:
                        self.repo.cleanup_worker_copy(worker_id)
                    except Exception:
                        pass

        self.metrics.active_workers = 0

    def _execute_worker_task(
        self, task: Task, worker_dir: Path, worker_id: str
    ) -> Handoff:
        """Execute a single worker task (runs in thread pool)."""
        return self.worker_agent.run_task(
            task=task,
            worker_dir=worker_dir,
            repo_manager=self.repo,
        )

    # ------------------------------------------------------------------
    # Handoff Routing
    # ------------------------------------------------------------------

    def _deliver_handoff(self, planner_id: str, handoff: Handoff):
        """Thread-safe delivery of a handoff to a planner's inbox."""
        with self._inbox_lock:
            if planner_id in self.handoff_inbox:
                self.handoff_inbox[planner_id].append(handoff)
            else:
                # Planner might have been removed; route to parent
                for pid, state in self.planners.items():
                    if planner_id in state.child_planner_ids:
                        self.handoff_inbox.setdefault(pid, []).append(handoff)
                        break

    def _drain_handoffs(self, planner_id: str) -> list[Handoff]:
        """Thread-safe retrieval of all pending handoffs for a planner."""
        with self._inbox_lock:
            handoffs = self.handoff_inbox.get(planner_id, [])
            self.handoff_inbox[planner_id] = []
        return handoffs

    # ------------------------------------------------------------------
    # PHASE 4: Freshness Phase
    # ------------------------------------------------------------------

    def _run_freshness_phase(self):
        """Run freshness checks on all active planners."""
        for state in self.planners.values():
            if state.is_complete:
                continue

            try:
                if self.freshness.should_summarize(state):
                    self.freshness.compress(state)
                elif self.freshness.should_refresh(state):
                    recent_info = f"Iteration {state.iteration_count}. " \
                                  f"Tasks emitted: {state.tasks_emitted}. " \
                                  f"Tasks completed: {state.tasks_completed}."
                    self.freshness.refresh(state, recent_info)
            except Exception as e:
                logger.warning(f"[FRESHNESS] Error for planner {state.id}: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        """Clean up all worker worktrees on shutdown."""
        worktrees_dir = Path(self.repo_path) / ".agent_worktrees"
        if worktrees_dir.exists():
            for item in worktrees_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(str(item), ignore_errors=True)
                except Exception:
                    pass
        logger.info("[CLEANUP] Removed all worker worktrees")


# ===========================================================================
# 6. DSPy EVALUATION METRICS (for optimization)
# ===========================================================================

def task_completion_metric(example, prediction, trace=None) -> float:
    """
    Metric for DSPy optimization: measures task completion quality.
    Used with dspy.BootstrapFewShot or dspy.MIPROv2.
    """
    score = 0.0

    # Did the system produce commits?
    if prediction.get("total_commits", 0) > 0:
        score += 0.3

    # Task completion rate
    created = prediction.get("total_tasks_created", 1)
    completed = prediction.get("total_tasks_completed", 0)
    score += 0.4 * (completed / max(created, 1))

    # Low error rate
    error_rate = prediction.get("error_rate", 1.0)
    score += 0.3 * (1.0 - min(error_rate, 1.0))

    return score


# ===========================================================================
# 7. ENTRY POINT
# ===========================================================================

def main():
    """Example usage of the multi-agent system."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Self-Driving Codebase")
    parser.add_argument("--repo", required=True, help="Path to the target repository")
    parser.add_argument("--instructions", required=True, help="What to build/do")
    parser.add_argument("--model", default="openai/meta-llama/Meta-Llama-3-8B-Instruct", help="LLM model to use")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent workers")
    parser.add_argument("--depth", type=int, default=3, help="Max planner recursion depth")
    parser.add_argument("--iterations", type=int, default=100, help="Max orchestration cycles")
    parser.add_argument("--api-key", default="", help="API key for the LLM provider")
    parser.add_argument("--api-base", default="http://localhost:7501/v1", help="API base URL")
    parser.add_argument("--model-type", default="chat", help="Model type (chat or text)")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens for LLM response")

    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=args.max_tokens, api_key=args.api_key, api_base=args.api_base, model_type=args.model_type)
    dspy.configure(lm=lm)

    # Run the system
    orchestrator = MultiAgentOrchestrator(
        max_workers=args.workers,
        max_depth=args.depth,
        max_iterations=args.iterations,
    )

    results = orchestrator.run(
        repo_path=args.repo,
        user_instructions=args.instructions,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()