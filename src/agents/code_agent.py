"""
Hierarchical Code Agent - Based on Cursor Blog Algorithm

Architecture:
- MainPlanner: Explores entire codebase, identifies major areas, spawns SubPlanners
- SubPlanner: Explores specific domain, creates atomic tasks, can spawn more SubPlanners
- Worker: Grabs one task, completes it fully, no coordination with others
- Judge: Reviews cycle completion, decides continue or stop

Flow:
LOOP:
  1. MainPlanner explores codebase and spawns SubPlanners
  2. SubPlanners (parallel) explore domains and create tasks
  3. Workers (concurrent) execute tasks independently
  4. Judge reviews and decides: continue or stop
  5. If continue, fresh start and LOOP again
"""

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Set
from pydantic import BaseModel

from src.ai import agent, step, LM
from src.logger.logger import get_logger

logger = get_logger(__name__, level='DEBUG')


# =============================================================================
# Data Models
# =============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """Atomic task unit for Workers"""
    id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    file_paths: List[str]
    created_by: str  # planner_id
    assigned_to: Optional[str] = None  # worker_id
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "file_paths": self.file_paths,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class PlannerContext:
    """Context for a planner (Main or Sub)"""
    planner_id: str
    domain: str  # e.g., "entire_codebase", "auth_system", "api_layer"
    file_patterns: List[str]  # e.g., ["src/auth/**/*.py"]
    parent_planner: Optional[str] = None
    sub_planners: List[str] = field(default_factory=list)
    tasks_created: List[str] = field(default_factory=list)


class CycleResult(BaseModel):
    """Result of a single cycle"""
    cycle_number: int
    tasks_completed: int
    tasks_failed: int
    sub_planners_spawned: int
    workers_active: int
    continue_next_cycle: bool
    judge_reasoning: str


# =============================================================================
# Database for Task Management
# =============================================================================

class TaskDatabase:
    """SQLite database for managing tasks across agents"""

    def __init__(self, db_path: str = "code_agent_tasks.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize task database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                file_paths TEXT NOT NULL,
                created_by TEXT NOT NULL,
                assigned_to TEXT,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        ''')

        # Planners table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS planners (
                planner_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                file_patterns TEXT NOT NULL,
                parent_planner TEXT,
                sub_planners TEXT,
                tasks_created TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        # Cycles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cycles (
                cycle_number INTEGER PRIMARY KEY,
                tasks_completed INTEGER,
                tasks_failed INTEGER,
                sub_planners_spawned INTEGER,
                workers_active INTEGER,
                continue_next_cycle BOOLEAN,
                judge_reasoning TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def add_task(self, task: Task):
        """Add a new task"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.id, task.title, task.description, task.priority.value,
            task.status.value, json.dumps(task.file_paths), task.created_by,
            task.assigned_to, task.result, task.error,
            task.created_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None
        ))
        conn.commit()
        conn.close()

    def get_pending_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get all pending tasks, ordered by priority"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT * FROM tasks
            WHERE status = ?
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END
        '''

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (TaskStatus.PENDING.value,))
        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            tasks.append(Task(
                id=row[0], title=row[1], description=row[2],
                priority=TaskPriority(row[3]), status=TaskStatus(row[4]),
                file_paths=json.loads(row[5]), created_by=row[6],
                assigned_to=row[7], result=row[8], error=row[9],
                created_at=datetime.fromisoformat(row[10]),
                completed_at=datetime.fromisoformat(row[11]) if row[11] else None
            ))
        return tasks

    def update_task_status(self, task_id: str, status: TaskStatus,
                          result: Optional[str] = None, error: Optional[str] = None):
        """Update task status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completed_at = datetime.now().isoformat() if status == TaskStatus.COMPLETED else None

        cursor.execute('''
            UPDATE tasks
            SET status = ?, result = ?, error = ?, completed_at = ?
            WHERE id = ?
        ''', (status.value, result, error, completed_at, task_id))

        conn.commit()
        conn.close()

    def assign_task(self, task_id: str, worker_id: str):
        """Assign task to worker"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tasks SET assigned_to = ?, status = ? WHERE id = ?
        ''', (worker_id, TaskStatus.IN_PROGRESS.value, task_id))
        conn.commit()
        conn.close()

    def get_cycle_stats(self) -> Dict:
        """Get statistics for current cycle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT status, COUNT(*) FROM tasks GROUP BY status')
        stats = dict(cursor.fetchall())

        conn.close()
        return stats

    def clear_tasks(self):
        """Clear all tasks (for new cycle)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tasks')
        conn.commit()
        conn.close()


# =============================================================================
# Code Analysis Tools
# =============================================================================

def list_files_recursive(root_path: str, pattern: str = "**/*.py") -> str:
    """
    List all files matching pattern recursively.

    Args:
        root_path: Root directory to search
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts")

    Returns:
        JSON string with file paths and basic stats
    """
    try:
        root = Path(root_path)
        files = list(root.glob(pattern))

        result = {
            "total_files": len(files),
            "files": [
                {
                    "path": str(f.relative_to(root)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                }
                for f in sorted(files)[:100]  # Limit to 100 files for sanity
            ]
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def read_file_content(file_path: str, start_line: Optional[int] = None,
                      end_line: Optional[int] = None) -> str:
    """
    Read file content, optionally with line range.

    Args:
        file_path: Path to file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)

    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line is not None and end_line is not None:
            lines = lines[start_line-1:end_line]
        elif start_line is not None:
            lines = lines[start_line-1:]

        content = ''.join(lines)

        return json.dumps({
            "path": file_path,
            "total_lines": len(lines),
            "content": content
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def write_file_content(file_path: str, content: str, create_backup: bool = True) -> str:
    """
    Write content to file.

    Args:
        file_path: Path to file
        content: Content to write
        create_backup: Whether to create .bak file

    Returns:
        Success message or error
    """
    try:
        path = Path(file_path)

        # Create backup if file exists
        if create_backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.bak')
            path.rename(backup_path)

        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return json.dumps({
            "success": True,
            "path": file_path,
            "bytes_written": len(content)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_code_pattern(root_path: str, pattern: str, file_extension: str = "py") -> str:
    """
    Search for code pattern in files.

    Args:
        root_path: Root directory to search
        pattern: Text pattern to search for
        file_extension: File extension to search in

    Returns:
        JSON with matches
    """
    try:
        root = Path(root_path)
        files = root.glob(f"**/*.{file_extension}")

        matches = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.lower() in line.lower():
                            matches.append({
                                "file": str(file_path.relative_to(root)),
                                "line": line_num,
                                "content": line.strip()
                            })
            except:
                continue

        return json.dumps({
            "pattern": pattern,
            "total_matches": len(matches),
            "matches": matches[:50]  # Limit to 50 matches
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_directory_structure(root_path: str, max_depth: int = 3) -> str:
    """
    Get directory structure as tree.

    Args:
        root_path: Root directory
        max_depth: Maximum depth to traverse

    Returns:
        Directory tree as string
    """
    try:
        root = Path(root_path)
        tree_lines = []

        def build_tree(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return

            try:
                entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    current_prefix = "└── " if is_last else "├── "
                    tree_lines.append(f"{prefix}{current_prefix}{entry.name}")

                    if entry.is_dir() and not entry.name.startswith('.'):
                        extension = "    " if is_last else "│   "
                        build_tree(entry, prefix + extension, depth + 1)
            except PermissionError:
                pass

        tree_lines.append(root.name)
        build_tree(root)

        return "\n".join(tree_lines)
    except Exception as e:
        return f"Error: {str(e)}"


def git_status() -> str:
    """Get git status"""
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--short'],
                              capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"


def git_commit_and_push(message: str, branch: str) -> str:
    """
    Commit and push changes to branch.

    Args:
        message: Commit message
        branch: Branch name

    Returns:
        Result message
    """
    try:
        import subprocess

        # Add all changes
        subprocess.run(['git', 'add', '-A'], check=True)

        # Commit
        result = subprocess.run(['git', 'commit', '-m', message],
                              capture_output=True, text=True)

        # Push
        push_result = subprocess.run(['git', 'push', '-u', 'origin', branch],
                                   capture_output=True, text=True)

        return json.dumps({
            "success": True,
            "commit": result.stdout,
            "push": push_result.stdout
        })
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "stderr": e.stderr if hasattr(e, 'stderr') else None
        })


# =============================================================================
# Agent Implementations
# =============================================================================

async def main_planner_agent(
    goal: str,
    root_path: str,
    lm: LM,
    db: TaskDatabase,
    cycle_number: int
) -> PlannerContext:
    """
    MainPlanner - Explores entire codebase and creates high-level plan.

    Responsibilities:
    - Understand the goal
    - Explore codebase structure
    - Identify major areas/domains
    - Spawn SubPlanners for each domain
    - Create overview context
    """

    planner_id = f"main_planner_cycle_{cycle_number}"

    system_prompt = f"""You are the MainPlanner in a hierarchical code agent system.

Your goal: {goal}

Your responsibilities:
1. Explore the entire codebase structure to understand what exists
2. Identify 2-5 major domains/areas that need work to achieve the goal
3. For each domain, specify:
   - Domain name (e.g., "auth_system", "api_layer", "database")
   - File patterns to focus on (e.g., ["src/auth/**/*.py"])
   - Brief description of what needs to be done
4. Create SubPlanner contexts for parallel exploration

You are in cycle {cycle_number}. The codebase is the source of truth - explore it fresh.

CRITICAL RULES:
- Use tools extensively to explore the codebase
- Start with directory structure, then dive into specific areas
- Don't make assumptions - verify with tools
- Focus on code organization, not detailed implementation
- Identify clear boundaries between domains
- Each domain should be independently explorable

Output format (JSON):
{{
  "domains": [
    {{
      "name": "domain_name",
      "file_patterns": ["pattern1", "pattern2"],
      "description": "What needs to be done in this domain",
      "priority": "critical|high|medium|low"
    }}
  ],
  "overview": "High-level summary of the plan"
}}
"""

    tools = [
        list_files_recursive,
        read_file_content,
        search_code_pattern,
        get_directory_structure,
    ]

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Explore the codebase at {root_path} and create a high-level plan."}
    ]

    logger.info(f"[MainPlanner] Starting exploration for cycle {cycle_number}")

    async for result in agent(
        lm=lm,
        history=history,
        tools=tools,
        max_iterations=15,
        logger=logger
    ):
        pass

    # Parse the final response to extract domains
    try:
        # Look for JSON in the response
        import re
        json_match = re.search(r'\{[\s\S]*\}', result.final_response)
        if json_match:
            plan_data = json.loads(json_match.group(0))
        else:
            # Fallback: create a single domain
            plan_data = {
                "domains": [{
                    "name": "main_implementation",
                    "file_patterns": ["src/**/*.py"],
                    "description": goal,
                    "priority": "high"
                }],
                "overview": result.final_response
            }
    except:
        plan_data = {
            "domains": [{
                "name": "main_implementation",
                "file_patterns": ["src/**/*.py"],
                "description": goal,
                "priority": "high"
            }],
            "overview": result.final_response
        }

    logger.info(f"[MainPlanner] Identified {len(plan_data['domains'])} domains")

    context = PlannerContext(
        planner_id=planner_id,
        domain="entire_codebase",
        file_patterns=["**/*"],
        sub_planners=[f"sub_planner_{d['name']}" for d in plan_data['domains']]
    )

    # Store the plan data for SubPlanners
    context.plan_data = plan_data

    return context


async def sub_planner_agent(
    domain_name: str,
    domain_description: str,
    file_patterns: List[str],
    priority: str,
    root_path: str,
    lm: LM,
    db: TaskDatabase,
    parent_planner: str,
    cycle_number: int
) -> PlannerContext:
    """
    SubPlanner - Explores specific domain and creates atomic tasks.

    Responsibilities:
    - Explore the assigned domain deeply
    - Understand existing code
    - Break work into atomic, executable tasks
    - Create detailed tasks for Workers
    - Can spawn more SubPlanners if domain is too complex
    """

    planner_id = f"sub_planner_{domain_name}_cycle_{cycle_number}"

    system_prompt = f"""You are a SubPlanner in a hierarchical code agent system.

Your domain: {domain_name}
Description: {domain_description}
File patterns: {file_patterns}
Priority: {priority}

Your responsibilities:
1. Explore your assigned domain in depth
2. Understand what code exists and what's missing
3. Break the work into 3-10 atomic tasks for Workers
4. Each task should be:
   - Completable by a single Worker independently
   - Focused on specific files (1-5 files max)
   - Clear and actionable
   - No dependencies on other tasks

You are in cycle {cycle_number}. Explore the current state of the code.

CRITICAL RULES:
- Use tools to read actual code in your domain
- Don't guess - verify what exists
- Tasks must be atomic and independent
- Each task should modify/create specific files
- Be specific about what needs to be done
- Consider edge cases and error handling

Output format (JSON):
{{
  "tasks": [
    {{
      "title": "Brief task title",
      "description": "Detailed description of what to do",
      "file_paths": ["path/to/file1.py", "path/to/file2.py"],
      "priority": "critical|high|medium|low"
    }}
  ],
  "spawn_sub_planners": [
    {{
      "subdomain_name": "name",
      "reason": "why this needs separate SubPlanner"
    }}
  ]
}}
"""

    tools = [
        list_files_recursive,
        read_file_content,
        search_code_pattern,
    ]

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Explore {domain_name} and create atomic tasks."}
    ]

    logger.info(f"[SubPlanner:{domain_name}] Starting domain exploration")

    async for result in agent(
        lm=lm,
        history=history,
        tools=tools,
        max_iterations=20,
        logger=logger
    ):
        pass

    # Parse tasks from response
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', result.final_response)
        if json_match:
            tasks_data = json.loads(json_match.group(0))
        else:
            tasks_data = {"tasks": [], "spawn_sub_planners": []}
    except:
        tasks_data = {"tasks": [], "spawn_sub_planners": []}

    # Create tasks in database
    task_ids = []
    for task_data in tasks_data.get("tasks", []):
        task_id = f"task_{domain_name}_{len(task_ids)}_{cycle_number}"
        task = Task(
            id=task_id,
            title=task_data.get("title", "Untitled task"),
            description=task_data.get("description", ""),
            priority=TaskPriority(task_data.get("priority", "medium")),
            status=TaskStatus.PENDING,
            file_paths=task_data.get("file_paths", []),
            created_by=planner_id
        )
        db.add_task(task)
        task_ids.append(task_id)

    logger.info(f"[SubPlanner:{domain_name}] Created {len(task_ids)} tasks")

    context = PlannerContext(
        planner_id=planner_id,
        domain=domain_name,
        file_patterns=file_patterns,
        parent_planner=parent_planner,
        tasks_created=task_ids
    )

    return context


async def worker_agent(
    task: Task,
    root_path: str,
    lm: LM,
    db: TaskDatabase,
    worker_id: str
) -> bool:
    """
    Worker - Executes a single atomic task.

    Responsibilities:
    - Grab one task from queue
    - Complete it fully
    - Make code changes
    - No coordination with other workers
    - Return success/failure
    """

    system_prompt = f"""You are a Worker in a hierarchical code agent system.

Your task: {task.title}
Description: {task.description}
Files to modify: {task.file_paths}
Priority: {task.priority.value}

Your responsibilities:
1. Read the relevant files
2. Understand the existing code
3. Make the required changes
4. Write the modified code
5. Verify the changes work

CRITICAL RULES:
- Focus ONLY on your assigned task
- Don't worry about other tasks or the big picture
- Complete the task fully before finishing
- Write clean, working code
- Handle errors gracefully
- Don't make assumptions - read the code first

You have these tools:
- read_file_content: Read files
- write_file_content: Write files
- search_code_pattern: Search for patterns
- list_files_recursive: List files

Complete the task and report back with:
- What you did
- What files you modified
- Any issues encountered
"""

    tools = [
        list_files_recursive,
        read_file_content,
        write_file_content,
        search_code_pattern,
    ]

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Complete the task: {task.title}\n\n{task.description}"}
    ]

    logger.info(f"[Worker:{worker_id}] Starting task {task.id}")

    try:
        async for result in agent(
            lm=lm,
            history=history,
            tools=tools,
            max_iterations=25,
            logger=logger
        ):
            pass

        # Update task as completed
        db.update_task_status(
            task.id,
            TaskStatus.COMPLETED,
            result=result.final_response
        )

        logger.info(f"[Worker:{worker_id}] Completed task {task.id}")
        return True

    except Exception as e:
        logger.error(f"[Worker:{worker_id}] Failed task {task.id}: {str(e)}")
        db.update_task_status(
            task.id,
            TaskStatus.FAILED,
            error=str(e)
        )
        return False


async def judge_agent(
    goal: str,
    cycle_stats: Dict,
    root_path: str,
    lm: LM,
    cycle_number: int,
    max_cycles: int
) -> CycleResult:
    """
    Judge - Reviews cycle completion and decides whether to continue.

    Responsibilities:
    - Review what was accomplished in this cycle
    - Check if goal is achieved
    - Decide: continue with fresh cycle or stop
    - Provide reasoning for decision
    """

    system_prompt = f"""You are the Judge in a hierarchical code agent system.

Original goal: {goal}

Cycle {cycle_number} of {max_cycles} statistics:
{json.dumps(cycle_stats, indent=2)}

Your responsibilities:
1. Review what was accomplished this cycle
2. Assess progress toward the goal
3. Check code quality and completeness
4. Decide whether to:
   - CONTINUE: Start a fresh cycle with new planning
   - STOP: Goal is achieved or max cycles reached

DECISION CRITERIA:
- Continue if: Goal not yet achieved, progress being made, cycles remaining
- Stop if: Goal achieved, no progress possible, max cycles reached

Be pragmatic. Fresh cycles help recover from mistakes and adapt to changes.

Output format (JSON):
{{
  "continue": true/false,
  "reasoning": "Detailed explanation of decision",
  "progress_percentage": 0-100,
  "key_achievements": ["achievement1", "achievement2"],
  "remaining_work": ["item1", "item2"]
}}
"""

    tools = [
        list_files_recursive,
        read_file_content,
        search_code_pattern,
        get_directory_structure,
    ]

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Review cycle {cycle_number} and decide whether to continue."}
    ]

    logger.info(f"[Judge] Reviewing cycle {cycle_number}")

    async for result in agent(
        lm=lm,
        history=history,
        tools=tools,
        max_iterations=10,
        logger=logger
    ):
        pass

    # Parse decision
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', result.final_response)
        if json_match:
            decision_data = json.loads(json_match.group(0))
        else:
            decision_data = {
                "continue": False,
                "reasoning": result.final_response,
                "progress_percentage": 50,
                "key_achievements": [],
                "remaining_work": []
            }
    except:
        decision_data = {
            "continue": False,
            "reasoning": result.final_response,
            "progress_percentage": 50,
            "key_achievements": [],
            "remaining_work": []
        }

    cycle_result = CycleResult(
        cycle_number=cycle_number,
        tasks_completed=cycle_stats.get(TaskStatus.COMPLETED.value, 0),
        tasks_failed=cycle_stats.get(TaskStatus.FAILED.value, 0),
        sub_planners_spawned=cycle_stats.get('sub_planners', 0),
        workers_active=cycle_stats.get('workers', 0),
        continue_next_cycle=decision_data.get("continue", False) and cycle_number < max_cycles,
        judge_reasoning=decision_data.get("reasoning", "No reasoning provided")
    )

    logger.info(f"[Judge] Decision: {'CONTINUE' if cycle_result.continue_next_cycle else 'STOP'}")
    logger.info(f"[Judge] Reasoning: {cycle_result.judge_reasoning}")

    return cycle_result


# =============================================================================
# Main Orchestrator
# =============================================================================

async def hierarchical_code_agent(
    goal: str,
    root_path: str,
    branch: str,
    lm: LM,
    max_cycles: int = 5,
    max_workers: int = 10,
    auto_commit: bool = True
) -> List[CycleResult]:
    """
    Main orchestrator for hierarchical code agent.

    Implements the Cursor blog algorithm:

    LOOP:
      1. MainPlanner explores codebase and spawns SubPlanners
      2. SubPlanners (parallel) explore domains and create tasks
      3. Workers (concurrent) execute tasks independently
      4. Judge reviews and decides: continue or stop
      5. If continue, fresh start and LOOP

    Args:
        goal: High-level goal to achieve
        root_path: Root path of codebase
        branch: Git branch to push to
        lm: Language model instance
        max_cycles: Maximum number of cycles to run
        max_workers: Maximum concurrent workers
        auto_commit: Whether to auto-commit changes after each cycle

    Returns:
        List of CycleResult objects
    """

    db = TaskDatabase()
    cycle_results = []

    logger.info("=" * 80)
    logger.info(f"Starting Hierarchical Code Agent")
    logger.info(f"Goal: {goal}")
    logger.info(f"Root: {root_path}")
    logger.info(f"Branch: {branch}")
    logger.info(f"Max Cycles: {max_cycles}")
    logger.info("=" * 80)

    for cycle_num in range(1, max_cycles + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"CYCLE {cycle_num}")
        logger.info(f"{'='*80}\n")

        # Clear tasks from previous cycle (fresh start)
        db.clear_tasks()

        # Step 1: MainPlanner explores and creates plan
        logger.info("[PHASE 1] MainPlanner exploring codebase...")
        main_context = await main_planner_agent(
            goal=goal,
            root_path=root_path,
            lm=lm,
            db=db,
            cycle_number=cycle_num
        )

        # Step 2: Spawn SubPlanners in parallel
        logger.info(f"[PHASE 2] Spawning {len(main_context.plan_data['domains'])} SubPlanners...")

        sub_planner_tasks = []
        for domain_data in main_context.plan_data['domains']:
            task_coro = sub_planner_agent(
                domain_name=domain_data['name'],
                domain_description=domain_data['description'],
                file_patterns=domain_data['file_patterns'],
                priority=domain_data.get('priority', 'medium'),
                root_path=root_path,
                lm=lm,
                db=db,
                parent_planner=main_context.planner_id,
                cycle_number=cycle_num
            )
            sub_planner_tasks.append(task_coro)

        sub_contexts = await asyncio.gather(*sub_planner_tasks)

        total_tasks = sum(len(ctx.tasks_created) for ctx in sub_contexts)
        logger.info(f"[PHASE 2] SubPlanners created {total_tasks} tasks")

        # Step 3: Workers execute tasks concurrently
        logger.info(f"[PHASE 3] Launching workers (max {max_workers} concurrent)...")

        pending_tasks = db.get_pending_tasks()

        async def worker_pool():
            """Worker pool that processes tasks concurrently"""
            workers_active = 0
            worker_counter = 0

            while True:
                # Get next pending task
                pending = db.get_pending_tasks(limit=1)
                if not pending:
                    break

                task = pending[0]
                worker_id = f"worker_{worker_counter}"
                worker_counter += 1

                # Assign task
                db.assign_task(task.id, worker_id)

                # Execute task
                await worker_agent(
                    task=task,
                    root_path=root_path,
                    lm=lm,
                    db=db,
                    worker_id=worker_id
                )

        # Run worker pool with limited concurrency
        worker_semaphore = asyncio.Semaphore(max_workers)

        async def bounded_worker():
            async with worker_semaphore:
                await worker_pool()

        # Launch workers
        await asyncio.gather(*[bounded_worker() for _ in range(min(max_workers, total_tasks))])

        logger.info(f"[PHASE 3] Workers completed all tasks")

        # Step 4: Optionally commit changes
        if auto_commit:
            logger.info("[PHASE 4] Committing changes...")
            commit_msg = f"Cycle {cycle_num}: {goal}"
            git_commit_and_push(commit_msg, branch)

        # Step 5: Judge reviews cycle
        logger.info("[PHASE 5] Judge reviewing cycle...")

        cycle_stats = db.get_cycle_stats()
        cycle_stats['sub_planners'] = len(sub_contexts)
        cycle_stats['workers'] = total_tasks

        cycle_result = await judge_agent(
            goal=goal,
            cycle_stats=cycle_stats,
            root_path=root_path,
            lm=lm,
            cycle_number=cycle_num,
            max_cycles=max_cycles
        )

        cycle_results.append(cycle_result)

        # Step 6: Decide whether to continue
        if not cycle_result.continue_next_cycle:
            logger.info(f"\n[JUDGE] Stopping after cycle {cycle_num}")
            logger.info(f"Reasoning: {cycle_result.judge_reasoning}")
            break

        logger.info(f"[JUDGE] Continuing to cycle {cycle_num + 1}")
        logger.info("Fresh start - planners will re-explore codebase\n")

    logger.info("\n" + "=" * 80)
    logger.info("Hierarchical Code Agent Complete")
    logger.info(f"Total Cycles: {len(cycle_results)}")
    logger.info(f"Total Tasks Completed: {sum(r.tasks_completed for r in cycle_results)}")
    logger.info("=" * 80)

    return cycle_results


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical Code Agent")
    parser.add_argument("--goal", required=True, help="Goal to achieve")
    parser.add_argument("--root-path", default=".", help="Root path of codebase")
    parser.add_argument("--branch", default="main", help="Git branch")
    parser.add_argument("--model", default="vllm:", help="LM model")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-cycles", type=int, default=5, help="Max cycles")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent workers")
    parser.add_argument("--no-auto-commit", action="store_true", help="Disable auto-commit")

    args = parser.parse_args()

    # Initialize LM
    lm = LM(model=args.model, api_base=args.api_base)

    # Run agent
    results = await hierarchical_code_agent(
        goal=args.goal,
        root_path=args.root_path,
        branch=args.branch,
        lm=lm,
        max_cycles=args.max_cycles,
        max_workers=args.max_workers,
        auto_commit=not args.no_auto_commit
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\nCycle {i}:")
        print(f"  Tasks Completed: {result.tasks_completed}")
        print(f"  Tasks Failed: {result.tasks_failed}")
        print(f"  SubPlanners: {result.sub_planners_spawned}")
        print(f"  Workers: {result.workers_active}")
        print(f"  Continue: {result.continue_next_cycle}")
        print(f"  Reasoning: {result.judge_reasoning[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
