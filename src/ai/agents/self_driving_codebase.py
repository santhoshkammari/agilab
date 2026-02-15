"""
Cursor's "Self-Driving Codebase" architecture in DSPy (minimal)

3 roles:
Planner  → breaks goal into tasks (recursive, can spawn sub-planners)
Worker   → codes one task, returns handoff note
Orchestrator → glues it together, feeds handoffs back up
"""

import dspy

# ── Configure your LM ──────────────────────────────────────────

lm = dspy.LM("openai/gpt-4o-mini")  # swap for any model
dspy.configure(lm=lm)

# ── Signatures (the "contracts") ────────────────────────────────

class Plan(dspy.Signature):
    """Break a goal into concrete, independent sub-tasks.
    If a task is still too large, mark it as needs_subplanner=True."""
    goal: str = dspy.InputField()
    codebase_state: str = dspy.InputField(desc="summary of current repo state")
    tasks: list[dict] = dspy.OutputField(
        desc="list of {task, needs_subplanner: bool}"
    )

class Code(dspy.Signature):
    """Implement a single focused task. Return code + handoff note."""
    task: str = dspy.InputField()
    context: str = dspy.InputField(desc="relevant files/code snippets")
    code: str = dspy.OutputField(desc="the code to commit")
    handoff: str = dspy.OutputField(
        desc="what was done, concerns, deviations, findings"
    )

# ── Modules ─────────────────────────────────────────────────────

class Planner(dspy.Module):
    """Owns a scope. Breaks it down. Spawns sub-planners or workers."""

    def __init__(self):
        self.plan = dspy.ChainOfThought(Plan)

    def forward(self, goal: str, codebase_state: str = "empty repo"):
        result = self.plan(goal=goal, codebase_state=codebase_state)
        return result.tasks

class Worker(dspy.Module):
    """Picks up ONE task. Codes it. Writes a handoff note back."""

    def __init__(self):
        self.code = dspy.ChainOfThought(Code)

    def forward(self, task: str, context: str = ""):
        result = self.code(task=task, context=context)
        return {"code": result.code, "handoff": result.handoff}

# ── Orchestrator (the glue) ─────────────────────────────────────

class SelfDrivingCodebase(dspy.Module):
    """
    Recursive orchestrator:
    1. Planner breaks goal → tasks
    2. Big tasks → sub-planner (recursive)
    3. Small tasks → worker
    4. Handoffs flow back up to planner for next iteration
    """

    def __init__(self, max_depth=2):
        self.planner = Planner()
        self.worker = Worker()
        self.max_depth = max_depth

    def forward(self, goal: str, codebase_state: str = "empty repo", depth: int = 0):
        # Step 1: Planner breaks goal into tasks
        tasks = self.planner(goal=goal, codebase_state=codebase_state)

        all_handoffs = []

        for t in tasks:
            task_desc = t.get("task", str(t))
            needs_sub = t.get("needs_subplanner", False)

            if needs_sub and depth < self.max_depth:
                # Step 2: Recursive sub-planner for big tasks
                print(f"{'  ' * depth}📋 Sub-planner: {task_desc[:60]}...")
                sub_handoffs = self.forward(
                    goal=task_desc,
                    codebase_state=codebase_state,
                    depth=depth + 1,
                )
                all_handoffs.extend(sub_handoffs)
            else:
                # Step 3: Worker codes it
                print(f"{'  ' * depth}🔨 Worker: {task_desc[:60]}...")
                result = self.worker(task=task_desc, context=codebase_state)
                all_handoffs.append(result)

        # Step 4: Handoffs flow back up (returned to parent planner)
        return all_handoffs

# ── Run it ──────────────────────────────────────────────────────

if __name__ == "__main__":
    system = SelfDrivingCodebase(max_depth=2)

    handoffs = system(
        goal="Build a simple HTTP server with routing and JSON responses",
        codebase_state="empty Python project with pyproject.toml",
    )

    print("\n" + "=" * 60)
    print(f"✅ Got {len(handoffs)} completed handoffs\n")
    for i, h in enumerate(handoffs, 1):
        print(f"── Handoff {i} ──")
        print(f"Handoff note: {h['handoff'][:200]}...")
        print()
