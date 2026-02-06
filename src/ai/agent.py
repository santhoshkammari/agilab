"""
Multi-agent orchestration system built on ai.Module and ai.Predict.

Inspired by Cursor's self-driving codebase architecture:
- Recursive planners decompose work into independent tasks
- Isolated workers execute tasks and produce structured handoffs
- No integrator bottleneck — workers submit directly
- Anti-fragile — individual failures don't crash the system

Usage:
    # Simple single-worker execution
    from ai.agent import run
    result = run("Summarize this document", tools=[read_file])

    # Full multi-agent orchestration
    from ai.agent import orchestrate
    handoffs = orchestrate("Refactor auth module", tools=[read_file, write_file], workers=5)
"""

import json
import asyncio
import uuid
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from .ai import LM, Predict, Module, Prediction


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Unit of work from Planner to Worker."""
    description: str
    context: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    created_by: str = ""
    metadata: dict = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Format task as a prompt string for the Worker LLM."""
        parts = [f"Task: {self.description}"]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.metadata:
            parts.append(f"Metadata: {json.dumps(self.metadata)}")
        return "\n".join(parts)


@dataclass
class Handoff:
    """Structured report from Worker back to Planner."""
    task_id: str
    worker_id: str
    status: TaskStatus
    summary: str = ""
    result: str = ""
    notes: str = ""
    concerns: str = ""
    deviations: str = ""
    findings: str = ""
    feedback: str = ""
    duration: float = 0.0
    error: Optional[str] = None


class EventType(str, Enum):
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    HANDOFF_RECEIVED = "handoff_received"
    PLAN_START = "plan_start"
    PLAN_COMPLETE = "plan_complete"
    REFLECT_START = "reflect_start"
    REFLECT_COMPLETE = "reflect_complete"
    WORKER_START = "worker_start"
    WORKER_COMPLETE = "worker_complete"
    ERROR = "error"


@dataclass
class AgentEvent:
    """Observable event for monitoring."""
    type: EventType
    agent_id: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class Scratchpad:
    """Rewritable working memory (freshness principle — overwrite, not append)."""

    def __init__(self):
        self._content: str = ""

    def write(self, content: str) -> None:
        """Overwrite scratchpad (NOT append)."""
        self._content = content

    def read(self) -> str:
        return self._content

    def clear(self) -> None:
        self._content = ""

    def __str__(self) -> str:
        return self._content

    def __bool__(self) -> bool:
        return bool(self._content)


# ---------------------------------------------------------------------------
# TaskQueue — async in-memory task management
# ---------------------------------------------------------------------------

class TaskQueue:
    """Async in-memory task queue with safe concurrent access."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._handoffs: dict[str, list[Handoff]] = {}  # planner_id -> handoffs
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def submit(self, task: Task) -> None:
        """Submit a task to the queue."""
        async with self._lock:
            self._tasks[task.id] = task
        await self._queue.put(task.id)

    async def claim(self) -> Optional[Task]:
        """Claim next pending task. Returns None if queue is empty."""
        try:
            task_id = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.IN_PROGRESS
            return task

    async def complete(self, handoff: Handoff) -> None:
        """Mark task as completed and store handoff."""
        async with self._lock:
            task = self._tasks.get(handoff.task_id)
            if task:
                task.status = handoff.status
            planner_id = task.created_by if task else ""
            if planner_id not in self._handoffs:
                self._handoffs[planner_id] = []
            self._handoffs[planner_id].append(handoff)

    async def fail(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.metadata["error"] = error

    def get_handoffs_for(self, planner_id: str) -> list[Handoff]:
        """Get all handoffs addressed to a specific planner."""
        return self._handoffs.get(planner_id, [])

    def clear_handoffs_for(self, planner_id: str) -> None:
        """Clear consumed handoffs."""
        self._handoffs.pop(planner_id, None)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def all_done(self) -> bool:
        return self._queue.empty() and all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self._tasks.values()
        )

    @property
    def tasks(self) -> dict[str, Task]:
        return dict(self._tasks)


# ---------------------------------------------------------------------------
# Planner — decomposes instructions into tasks, does NO execution
# ---------------------------------------------------------------------------

PLANNER_DECOMPOSE_SYSTEM = """\
You are a planning agent. Your ONLY job is to decompose instructions into \
independent, self-contained tasks.

Constraints:
- Each task MUST be self-contained — a worker with no memory of other tasks \
can execute it.
- No task should depend on the output of another task unless absolutely \
necessary.
- Describe WHAT needs to be done, not HOW.
- Return valid JSON: a list of objects with "description" and optional \
"context", "priority" (int, higher = more urgent), and "metadata" (dict).

Example output:
[
  {"description": "Extract all API endpoint definitions from auth.py", "priority": 2},
  {"description": "List all database models related to user sessions", "priority": 1}
]
"""

PLANNER_REFLECT_SYSTEM = """\
You are a planning agent reviewing completed work. Given the handoff reports \
from workers, decide if follow-up tasks are needed.

Constraints:
- Only create follow-up tasks if there are unresolved concerns, deviations, or \
incomplete work.
- Each follow-up MUST be self-contained.
- If all work is satisfactory, return an empty list: []
- Return valid JSON: a list of objects with "description" and optional \
"context", "priority", "metadata".
"""

PLANNER_SUBPLAN_SYSTEM = """\
You are a planning agent. Given a task description, decide whether it is \
complex enough to require further decomposition by a sub-planner.

Respond with JSON: {"needs_subplanner": true/false, "reason": "..."}
"""


class Planner(Module):
    """Decomposes instructions into tasks. Does NO coding or execution.

    Can recursively spawn sub-planners for complex tasks up to max_depth.
    """

    def __init__(
        self,
        planner_id: Optional[str] = None,
        max_depth: int = 2,
        depth: int = 0,
        lm: Optional[LM] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ):
        super().__init__()
        self.planner_id = planner_id or f"planner-{uuid.uuid4().hex[:8]}"
        self.max_depth = max_depth
        self.depth = depth
        self.on_event = on_event
        self.scratchpad = Scratchpad()

        self.decompose = Predict(
            "instructions, context -> tasks_json",
            system=PLANNER_DECOMPOSE_SYSTEM,
            lm=lm,
        )
        self.reflect = Predict(
            "handoffs, context -> follow_up_tasks_json",
            system=PLANNER_REFLECT_SYSTEM,
            lm=lm,
        )
        self.should_subplan = Predict(
            "task_description -> needs_subplanner, reason",
            system=PLANNER_SUBPLAN_SYSTEM,
            lm=lm,
        )

    def _emit(self, event_type: EventType, data: Any = None) -> None:
        if self.on_event:
            self.on_event(AgentEvent(type=event_type, agent_id=self.planner_id, data=data))

    def _parse_tasks_json(self, raw: str) -> list[dict]:
        """Extract JSON list from LLM output (tolerant of markdown fences)."""
        text = raw.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            # Try to find a JSON array in the text
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return []
            return []

    def forward(self, instructions: str, queue: TaskQueue, context: str = "") -> list[Task]:
        """Decompose instructions into tasks and submit them to the queue.

        Runs async internally but presents a sync interface (same pattern as Predict).
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use aforward() from async context")
        except RuntimeError as e:
            if "Use aforward()" in str(e):
                raise
            return asyncio.run(self.aforward(instructions, queue, context))

    async def aforward(self, instructions: str, queue: TaskQueue, context: str = "") -> list[Task]:
        """Async decompose instructions into tasks and submit to queue."""
        self._emit(EventType.PLAN_START, {"instructions": instructions})
        self.scratchpad.write(f"Planning: {instructions}")

        prediction = await self.decompose(
            instructions=instructions,
            context=context or "No additional context.",
        )

        raw = str(prediction)
        task_dicts = self._parse_tasks_json(raw)

        tasks: list[Task] = []
        for td in task_dicts:
            if not td.get("description"):
                continue
            task = Task(
                description=td["description"],
                context=td.get("context", ""),
                priority=td.get("priority", 0),
                created_by=self.planner_id,
                metadata=td.get("metadata", {}),
            )

            # Check if sub-planning is needed (and depth allows)
            if self.depth < self.max_depth:
                sub_pred = await self.should_subplan(
                    task_description=task.description
                )
                sub_raw = str(sub_pred)
                try:
                    sub_json = json.loads(sub_raw.strip())
                    if not isinstance(sub_json, dict):
                        sub_json = {}
                except json.JSONDecodeError:
                    # Try to extract JSON
                    start = sub_raw.find("{")
                    end = sub_raw.rfind("}")
                    if start != -1 and end != -1:
                        try:
                            sub_json = json.loads(sub_raw[start:end + 1])
                        except json.JSONDecodeError:
                            sub_json = {}
                    else:
                        sub_json = {}

                needs_sub = sub_json.get("needs_subplanner", False)
                if needs_sub is True or (isinstance(needs_sub, str) and needs_sub.lower() == "true"):
                    child_planner = Planner(
                        max_depth=self.max_depth,
                        depth=self.depth + 1,
                        lm=self.decompose.lm,
                        on_event=self.on_event,
                    )
                    sub_tasks = await child_planner.aforward(
                        task.description, queue, context=task.context
                    )
                    tasks.extend(sub_tasks)
                    self._emit(EventType.TASK_CREATED, {"sub_tasks": len(sub_tasks), "parent": task.description})
                    continue

            await queue.submit(task)
            tasks.append(task)
            self._emit(EventType.TASK_CREATED, {"task_id": task.id, "description": task.description})

        self._emit(EventType.PLAN_COMPLETE, {"task_count": len(tasks)})
        return tasks

    async def receive_handoffs(self, handoffs: list[Handoff], queue: TaskQueue, context: str = "") -> list[Task]:
        """Review handoffs and create follow-up tasks if needed."""
        if not handoffs:
            return []

        self._emit(EventType.REFLECT_START, {"handoff_count": len(handoffs)})

        handoff_summaries = []
        for h in handoffs:
            summary = {
                "task_id": h.task_id,
                "status": h.status.value if isinstance(h.status, Enum) else h.status,
                "summary": h.summary,
                "concerns": h.concerns,
                "deviations": h.deviations,
                "findings": h.findings,
                "feedback": h.feedback,
            }
            if h.error:
                summary["error"] = h.error
            handoff_summaries.append(summary)

        prediction = await self.reflect(
            handoffs=json.dumps(handoff_summaries, indent=2),
            context=context or "No additional context.",
        )

        raw = str(prediction)
        follow_up_dicts = self._parse_tasks_json(raw)

        follow_ups: list[Task] = []
        for td in follow_up_dicts:
            if not td.get("description"):
                continue
            task = Task(
                description=td["description"],
                context=td.get("context", ""),
                priority=td.get("priority", 0),
                created_by=self.planner_id,
                metadata=td.get("metadata", {}),
            )
            await queue.submit(task)
            follow_ups.append(task)
            self._emit(EventType.TASK_CREATED, {"task_id": task.id, "description": task.description, "follow_up": True})

        self._emit(EventType.REFLECT_COMPLETE, {"follow_up_count": len(follow_ups)})
        return follow_ups


# ---------------------------------------------------------------------------
# Worker — picks up a task, executes, produces a Handoff
# ---------------------------------------------------------------------------

WORKER_EXECUTE_SYSTEM = """\
You are a worker agent. Focus ONLY on the assigned task. Execute it thoroughly.

Constraints:
- Do not attempt tasks beyond your assignment.
- If you cannot complete the task, explain why clearly.
- Be precise and thorough in your output.
"""

WORKER_REPORT_SYSTEM = """\
You are a worker agent writing a structured handoff report. Given the task \
and your execution result, produce a report with these exact fields as JSON:

{
  "summary": "Brief summary of what was accomplished",
  "notes": "Any additional notes for the planner",
  "concerns": "Any concerns or risks identified",
  "deviations": "Any deviations from the original task",
  "findings": "Unexpected findings during execution",
  "feedback": "Suggestions for improving the task description"
}
"""


class Worker(Module):
    """Picks up a task, executes it, and produces a structured Handoff.

    Completely isolated: no reference to queue, planner, or other workers.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        lm: Optional[LM] = None,
        tools: Optional[list[Callable]] = None,
        timeout: float = 300.0,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ):
        super().__init__()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.timeout = timeout
        self.on_event = on_event
        self.scratchpad = Scratchpad()

        self.execute = Predict(
            "task, context -> result",
            system=WORKER_EXECUTE_SYSTEM,
            lm=lm,
            tools=tools,
        )
        self.report = Predict(
            "task, result -> report_json",
            system=WORKER_REPORT_SYSTEM,
            lm=lm,
        )

    def _emit(self, event_type: EventType, data: Any = None) -> None:
        if self.on_event:
            self.on_event(AgentEvent(type=event_type, agent_id=self.worker_id, data=data))

    def _parse_report(self, raw: str) -> dict:
        """Parse structured report JSON from LLM output."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {"summary": text}

    def forward(self, task: Task) -> Handoff:
        """Execute task and return Handoff. Sync wrapper."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use aforward() from async context")
        except RuntimeError as e:
            if "Use aforward()" in str(e):
                raise
            return asyncio.run(self.aforward(task))

    async def aforward(self, task: Task) -> Handoff:
        """Async: execute task and return structured Handoff."""
        self._emit(EventType.WORKER_START, {"task_id": task.id, "description": task.description})
        self.scratchpad.write(f"Working on: {task.description}")
        start = time.time()

        try:
            exec_result = await asyncio.wait_for(
                self.execute(task=task.to_prompt(), context=task.context or "No additional context."),
                timeout=self.timeout,
            )
            result_text = str(exec_result)

            report_pred = await self.report(task=task.to_prompt(), result=result_text)
            report = self._parse_report(str(report_pred))

            duration = time.time() - start
            handoff = Handoff(
                task_id=task.id,
                worker_id=self.worker_id,
                status=TaskStatus.COMPLETED,
                summary=report.get("summary", result_text[:200]),
                result=result_text,
                notes=report.get("notes", ""),
                concerns=report.get("concerns", ""),
                deviations=report.get("deviations", ""),
                findings=report.get("findings", ""),
                feedback=report.get("feedback", ""),
                duration=duration,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start
            handoff = Handoff(
                task_id=task.id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                summary="Task timed out",
                error=f"Execution exceeded {self.timeout}s timeout",
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start
            handoff = Handoff(
                task_id=task.id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                summary=f"Task failed: {e}",
                error=str(e),
                duration=duration,
            )

        self._emit(
            EventType.WORKER_COMPLETE,
            {"task_id": task.id, "status": handoff.status.value, "duration": duration},
        )
        return handoff


# ---------------------------------------------------------------------------
# Orchestrator — top-level coordinator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Top-level coordinator wiring Planner → TaskQueue → Workers → Handoffs → Planner.

    Anti-fragile: timeouts, retries, individual failures don't crash system.
    Observable: on_event callback for monitoring.

    Usage:
        orch = Orchestrator(tools=[read_file, write_file], max_workers=4)
        handoffs = orch("Refactor the auth module to use JWT")
    """

    def __init__(
        self,
        lm: Optional[LM] = None,
        tools: Optional[list[Callable]] = None,
        max_workers: int = 3,
        max_rounds: int = 3,
        max_retries: int = 2,
        worker_timeout: float = 300.0,
        planner_max_depth: int = 2,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ):
        self.lm = lm
        self.tools = tools
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.max_retries = max_retries
        self.worker_timeout = worker_timeout
        self.planner_max_depth = planner_max_depth
        self.on_event = on_event
        self.queue = TaskQueue()
        self.all_handoffs: list[Handoff] = []

    def _emit(self, event_type: EventType, data: Any = None) -> None:
        if self.on_event:
            self.on_event(AgentEvent(type=event_type, agent_id="orchestrator", data=data))

    async def _run_worker(self, worker: Worker, task: Task, retries: int = 0) -> Handoff:
        """Run a single worker on a task with retries."""
        handoff = await worker.aforward(task)

        if handoff.status == TaskStatus.FAILED and retries < self.max_retries:
            self._emit(EventType.ERROR, {
                "task_id": task.id,
                "error": handoff.error,
                "retry": retries + 1,
            })
            return await self._run_worker(worker, task, retries + 1)

        return handoff

    async def _fan_out_workers(self) -> list[Handoff]:
        """Claim all pending tasks and fan out to workers concurrently."""
        tasks: list[Task] = []
        while True:
            task = await self.queue.claim()
            if task is None:
                break
            tasks.append(task)

        if not tasks:
            return []

        async def process(task: Task) -> Handoff:
            worker = Worker(
                lm=self.lm,
                tools=self.tools,
                timeout=self.worker_timeout,
                on_event=self.on_event,
            )
            handoff = await self._run_worker(worker, task)
            await self.queue.complete(handoff)
            return handoff

        # Run workers concurrently, bounded by max_workers
        sem = asyncio.Semaphore(self.max_workers)

        async def bounded(task: Task) -> Handoff:
            async with sem:
                return await process(task)

        handoffs = await asyncio.gather(
            *(bounded(t) for t in tasks),
            return_exceptions=True,
        )

        results: list[Handoff] = []
        for i, h in enumerate(handoffs):
            if isinstance(h, Exception):
                error_handoff = Handoff(
                    task_id=tasks[i].id,
                    worker_id="error",
                    status=TaskStatus.FAILED,
                    summary=f"Unexpected error: {h}",
                    error=str(h),
                )
                await self.queue.complete(error_handoff)
                results.append(error_handoff)
            else:
                results.append(h)

        return results

    async def arun(self, instructions: str, context: str = "") -> list[Handoff]:
        """Async orchestration loop.

        1. Planner decomposes instructions into tasks
        2. Workers execute concurrently
        3. Planner reflects on handoffs, maybe creates follow-ups
        4. Loop until no follow-ups or max_rounds reached
        """
        planner = Planner(
            max_depth=self.planner_max_depth,
            lm=self.lm,
            on_event=self.on_event,
        )

        # Step 1: Initial planning
        await planner.aforward(instructions, self.queue, context)

        for round_num in range(self.max_rounds):
            # Step 2: Fan out workers
            handoffs = await self._fan_out_workers()
            self.all_handoffs.extend(handoffs)

            for h in handoffs:
                self._emit(EventType.HANDOFF_RECEIVED, {
                    "task_id": h.task_id,
                    "status": h.status.value if isinstance(h.status, Enum) else h.status,
                    "summary": h.summary,
                })

            # Step 3: Planner reflects
            follow_ups = await planner.receive_handoffs(handoffs, self.queue, context)

            # Step 4: If no follow-ups, we're done
            if not follow_ups:
                break

        return self.all_handoffs

    def run(self, instructions: str, context: str = "") -> list[Handoff]:
        """Sync wrapper for arun."""
        return asyncio.run(self.arun(instructions, context))

    def __call__(self, instructions: str, context: str = "") -> list[Handoff]:
        """Call orchestrator (sync or async based on context)."""
        try:
            asyncio.get_running_loop()
            return self.arun(instructions, context)
        except RuntimeError:
            return self.run(instructions, context)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def _single_worker_run(
    instructions: str,
    lm: Optional[LM] = None,
    tools: Optional[list[Callable]] = None,
    context: str = "",
    timeout: float = 300.0,
) -> Handoff:
    """Run a single worker on a task (no planner, no queue)."""
    task = Task(description=instructions, context=context, created_by="direct")
    worker = Worker(lm=lm, tools=tools, timeout=timeout)
    return await worker.aforward(task)


def run(
    instructions: str,
    lm: Optional[LM] = None,
    tools: Optional[list[Callable]] = None,
    context: str = "",
    timeout: float = 300.0,
) -> Handoff:
    """Simplest API: single worker, no planner.

    Usage:
        from ai.agent import run
        result = run("Summarize this document", tools=[read_file])
        print(result.summary)
    """
    try:
        asyncio.get_running_loop()
        return _single_worker_run(instructions, lm, tools, context, timeout)
    except RuntimeError:
        return asyncio.run(_single_worker_run(instructions, lm, tools, context, timeout))


def orchestrate(
    instructions: str,
    lm: Optional[LM] = None,
    tools: Optional[list[Callable]] = None,
    context: str = "",
    workers: int = 3,
    max_rounds: int = 3,
    max_retries: int = 2,
    worker_timeout: float = 300.0,
    planner_max_depth: int = 2,
    on_event: Optional[Callable[[AgentEvent], None]] = None,
) -> list[Handoff]:
    """Full multi-agent orchestration.

    Usage:
        from ai.agent import orchestrate
        handoffs = orchestrate(
            "Refactor auth module",
            tools=[read_file, write_file],
            workers=5,
        )
        for h in handoffs:
            print(h.summary)
    """
    orch = Orchestrator(
        lm=lm,
        tools=tools,
        max_workers=workers,
        max_rounds=max_rounds,
        max_retries=max_retries,
        worker_timeout=worker_timeout,
        planner_max_depth=planner_max_depth,
        on_event=on_event,
    )
    return orch(instructions, context)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "Task",
    "TaskStatus",
    "Handoff",
    "AgentEvent",
    "EventType",
    "Scratchpad",
    "TaskQueue",
    "Planner",
    "Worker",
    "Orchestrator",
    "run",
    "orchestrate",
]
