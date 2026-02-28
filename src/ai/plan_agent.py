"""
Recursive parallel planning agents — Cursor-style final design.

Strict role separation:
  Planner  — pure LLM, no tools, only plans and delegates. Never codes.
  Worker   — tools only, no awareness of the larger system. Just executes.

Flow:
  Planner → decides SPLIT or WORKER
  SPLIT   → spawns SubPlanners in parallel (recursive)
  WORKER  → spawns WorkerAgent with focused task
  Handoffs bubble back up → planner receives them, loops, keeps planning.
"""
import json
import asyncio
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import ai
from file_tools import tools as file_tools

SCRATCHPAD = Path("scratchpad.md")


def git_commit(files: list[str], message: str) -> str:
    """Commit changed files to git. Returns commit hash or error."""
    if not files:
        return "no files to commit"
    try:
        subprocess.run(["git", "add"] + files, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            check=True, capture_output=True, text=True
        )
        # Extract short hash from output
        line = result.stdout.strip().split('\n')[0]
        return line
    except subprocess.CalledProcessError as e:
        return f"git error: {e.stderr.strip()}"


# --- AI Config ---
import aiohttp
lm = ai.LM(
    api_base="http://192.168.170.76:8000",
    timeout=aiohttp.ClientTimeout(total=None, connect=120, sock_read=None),
)
ai.configure(lm)
# -----------------

# --- System Prompts ---

PLANNER_SYSTEM = """You are a planning agent. You NEVER write code or use tools yourself.
Your only job is to break down tasks and delegate.

Actions:
- SPLIT: task has 2+ INDEPENDENT parts that can run in parallel
- WORKER: task is sequential, dependent, or a single atomic action

Before deciding, check: does step B need step A's OUTPUT to begin?
- If yes → WORKER (a worker can do multiple sequential steps)
- If no → SPLIT

Rules:
- subtasks must be valid JSON array of strings when action is SPLIT
- Set subtasks to "[]" when action is WORKER
- worker_task must be a clear, self-contained instruction when action is WORKER
- Maximum 2-4 subtasks per SPLIT
"""

WORKER_SYSTEM = """You are a worker agent. Execute ONLY the exact task given. Nothing more, nothing less.

Constraints:
- Do ONLY what the task says. Do not analyze, verify, or extend beyond scope.
- ALWAYS use tools. Never make up or guess results.
- No follow-up work. No suggestions for improvements. Just execute and report.

When done, output a JSON handoff:
{
  "what_done": "what you accomplished",
  "findings": "unexpected discoveries ONLY (empty string if nothing unexpected)",
  "concerns": "blockers or errors encountered (empty string if none)",
  "files_changed": ["files", "you", "modified"]
}
"""

LOOP_SYSTEM = """You are a planning agent reviewing handoffs from completed work.
Decide if the ORIGINAL task requires more work based ONLY on real blockers/concerns.

Rules:
- DONE: original goal is achieved, even if imperfectly. Default to DONE.
- SPLIT: only if concerns reveal NEW independent work that blocks the goal
- WORKER: only if a specific concern must be fixed before the goal is met
- Suggestions and nice-to-haves are NOT reasons to continue. Ignore them.
- If concerns are minor or cosmetic, choose DONE.
"""


# --- Handoff dataclass ---

@dataclass
class Handoff:
    agent: str
    task: str
    what_done: str = ""
    findings: str = ""
    concerns: str = ""
    files_changed: list = field(default_factory=list)

    def to_str(self) -> str:
        return (
            f"TASK: {self.task}\n"
            f"DONE: {self.what_done}\n"
            f"FINDINGS: {self.findings}\n"
            f"CONCERNS: {self.concerns}\n"
            f"FILES: {', '.join(self.files_changed) or 'none'}"
        )

    @classmethod
    def from_json(cls, agent: str, task: str, raw: str) -> "Handoff":
        """Parse worker JSON output into structured Handoff."""
        try:
            # Try to extract JSON from the raw string
            start = raw.find('{')
            end = raw.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                files = data.get('files_changed', [])
                if isinstance(files, str):
                    files = [f.strip() for f in files.split(',') if f.strip()]
                return cls(
                    agent=agent,
                    task=task,
                    what_done=data.get('what_done', raw),
                    findings=data.get('findings', ''),
                    concerns=data.get('concerns', ''),
                    files_changed=files,
                )
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: treat raw as what_done
        return cls(agent=agent, task=task, what_done=raw)


# --- Worker: tools only, no planning awareness ---

class Worker:
    def __init__(self, task: str, tools=None, verbose: bool = False, name: str = "Worker"):
        self.task = task
        self.name = name
        self.verbose = verbose
        self._executor = ai.Predict(
            "task -> handoff",
            system=WORKER_SYSTEM,
            tools=tools or file_tools,
            verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(f"    [{self.name}] {msg}", file=sys.stderr, flush=True)

    async def run(self) -> Handoff:
        self._log(f"executing: {self.task}")
        try:
            result = await self._executor._arun(task=self.task)
            raw = getattr(result, 'handoff', str(result))
            handoff = Handoff.from_json(self.name, self.task, raw)
        except Exception as e:
            self._log(f"error: {e}")
            handoff = Handoff(agent=self.name, task=self.task, what_done="Failed", concerns=str(e))

        # #4 Git commit if files were changed
        if handoff.files_changed:
            commit_msg = f"[{self.name}] {self.task[:60]}"
            result = git_commit(handoff.files_changed, commit_msg)
            self._log(f"git: {result}")

        self._log(f"done: {handoff.what_done[:80]}")
        return handoff


# --- Planner: pure LLM, no tools, recursive + continuous loop ---

class Planner:
    def __init__(
        self,
        task: str,
        depth: int = 0,
        max_depth: int = 3,
        max_loops: int = 5,
        tools=None,
        verbose: bool = False,
        name: Optional[str] = None,
    ):
        self.task = task
        self.depth = depth
        self.max_depth = max_depth
        self.max_loops = max_loops
        self.tools = tools or file_tools
        self.verbose = verbose
        self.name = name or f"Planner(d={depth})"

        # Pure LLM — no tools
        self._decide = ai.Predict(
            "task, depth, context -> action, subtasks, worker_task, reason",
            system=PLANNER_SYSTEM,
            verbose=verbose,
        )
        # Loop: receives handoffs, decides next action
        self._loop = ai.Predict(
            "original_task, handoffs, loop -> action, subtasks, worker_task, reason",
            system=LOOP_SYSTEM,
            verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            indent = "  " * self.depth
            print(f"{indent}[{self.name}] {msg}", file=sys.stderr, flush=True)

    def _handoffs_to_context(self, handoffs: list[Handoff]) -> str:
        return "\n\n".join(h.to_str() for h in handoffs)

    def _read_scratchpad(self) -> str:
        """#5 Read planner's scratchpad (only root planner uses global scratchpad)."""
        if self.depth == 0 and SCRATCHPAD.exists():
            return SCRATCHPAD.read_text()
        return ""

    def _write_scratchpad(self, handoffs: list[Handoff], loop: int):
        """#5 Rewrite scratchpad with current state — never append, always rewrite."""
        if self.depth != 0:
            return  # Only root planner writes global scratchpad
        concerns = [h.concerns for h in handoffs if h.concerns]
        content = (
            f"# Scratchpad — {self.name}\n"
            f"## Task\n{self.task}\n\n"
            f"## Loop {loop} — {len(handoffs)} handoffs received\n"
            + self._handoffs_to_context(handoffs)
            + (f"\n\n## Active Concerns\n" + "\n".join(f"- {c}" for c in concerns) if concerns else "\n\n## No concerns")
        )
        SCRATCHPAD.write_text(content)
        self._log(f"scratchpad rewritten (loop {loop})")

    async def run(self, context: str = "") -> Handoff:
        self._log(f"task: {self.task}")

        # #5 Load scratchpad as additional context (root planner only)
        scratch = self._read_scratchpad()
        if scratch:
            context = f"{context}\n\n## Previous State\n{scratch}".strip()

        # Initial decision
        decision = await self._decide._arun(
            task=self.task,
            depth=str(self.depth),
            context=context,
        )

        action = getattr(decision, 'action', 'WORKER').strip().upper()
        reason = getattr(decision, 'reason', '')
        self._log(f"decision: {action} — {reason}")

        # Execute first round
        handoffs = await self._execute(action, decision, context)

        # Only root planner (depth=0) loops. Subplanners are single-shot.
        if self.depth == 0:
            self._write_scratchpad(handoffs, 0)

            # No concerns = done immediately, no LLM call needed
            concerns = [h.concerns for h in handoffs if h.concerns]
            if not concerns:
                self._log("no concerns, done")
            else:
                # Concerns exist — loop and re-evaluate
                for loop_i in range(1, self.max_loops):
                    self._log(f"loop {loop_i}: {len(concerns)} concern(s), re-evaluating")
                    handoff_ctx = self._handoffs_to_context(handoffs)

                    next_decision = await self._loop._arun(
                        original_task=self.task,
                        handoffs=handoff_ctx,
                        loop=str(loop_i),
                    )

                    next_action = getattr(next_decision, 'action', 'DONE').strip().upper()
                    next_reason = getattr(next_decision, 'reason', '')
                    self._log(f"loop {loop_i} decision: {next_action} — {next_reason}")

                    if next_action == "DONE":
                        break

                    # More work needed
                    new_handoffs = await self._execute(next_action, next_decision, handoff_ctx)
                    handoffs.extend(new_handoffs)
                    self._write_scratchpad(handoffs, loop_i)

                    # Re-check concerns
                    concerns = [h.concerns for h in new_handoffs if h.concerns]
                    if not concerns:
                        self._log(f"loop {loop_i}: concerns resolved, done")
                        break

        return self._aggregate(handoffs)

    async def _execute(self, action: str, decision, context: str) -> list[Handoff]:
        """Execute a SPLIT or WORKER decision, return list of handoffs."""
        if action == "SPLIT" and self.depth < self.max_depth:
            return await self._split(decision, context)
        else:
            h = await self._delegate_worker(decision)
            return [h]

    async def _split(self, decision, context: str) -> list[Handoff]:
        """Spawn subplanners in parallel."""
        try:
            raw = getattr(decision, 'subtasks', '[]')
            subtasks = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._log("subtasks parse failed, falling back to WORKER")
            h = await self._delegate_worker(decision)
            return [h]

        if not subtasks:
            h = await self._delegate_worker(decision)
            return [h]

        self._log(f"spawning {len(subtasks)} subplanners in parallel")
        subplanners = [
            Planner(
                task=t,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                max_loops=self.max_loops,
                tools=self.tools,
                verbose=self.verbose,
                name=f"Planner(d={self.depth+1},#{i})",
            )
            for i, t in enumerate(subtasks)
        ]
        return list(await asyncio.gather(*[p.run(context) for p in subplanners]))

    async def _delegate_worker(self, decision) -> Handoff:
        """Spawn a single worker for this atomic task."""
        worker_task = getattr(decision, 'worker_task', '').strip() or self.task
        worker = Worker(
            task=worker_task,
            tools=self.tools,
            verbose=self.verbose,
            name=f"Worker(d={self.depth})",
        )
        return await worker.run()

    def _aggregate(self, handoffs: list[Handoff]) -> Handoff:
        """Merge all handoffs into a single parent handoff."""
        all_done = "\n".join(f"- [{h.agent}] {h.what_done}" for h in handoffs)
        all_findings = "\n".join(f"- [{h.agent}] {h.findings}" for h in handoffs if h.findings)
        all_concerns = "\n".join(f"- [{h.agent}] {h.concerns}" for h in handoffs if h.concerns)
        all_files = [f for h in handoffs for f in h.files_changed]
        self._log("aggregating all handoffs")
        return Handoff(
            agent=self.name,
            task=self.task,
            what_done=all_done,
            findings=all_findings,
            concerns=all_concerns,
            files_changed=all_files,
        )


# --- Entry point ---

async def run_plan(task: str, verbose: bool = True, max_depth: int = 3, max_loops: int = 5, tools=None):
    # Fresh run — clear stale scratchpad
    if SCRATCHPAD.exists():
        SCRATCHPAD.unlink()
    root = Planner(
        task=task,
        depth=0,
        max_depth=max_depth,
        max_loops=max_loops,
        tools=tools,
        verbose=verbose,
        name="RootPlanner",
    )
    return await root.run()


if __name__ == "__main__":
    task = "List all python files in current directory and read the content of ai.py"
    task = "websearch for today's ai news and fetch each and then summarize it"
    result = asyncio.run(run_plan(task, verbose=True))
    print("\n=== FINAL RESULT ===")
    print(result.to_str())
