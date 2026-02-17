# Multi-Agent Self-Driving Codebase (DSPy)

Implementation of Cursor's research on autonomous multi-agent coding systems.

## Architecture

```
User Instructions + Repo Path
            │
            ▼
    ┌───────────────────┐
    │  ORCHESTRATOR      │  ← Continuous loop: plan → execute → route → refresh
    │  (MultiAgent       │
    │   Orchestrator)    │
    └───────┬───────────┘
            │
    ╔═══════╧════════════════════════════════════════╗
    ║           PLANNER TREE (recursive)             ║
    ║                                                ║
    ║   ┌─────────────┐                              ║
    ║   │ ROOT PLANNER │ depth=0, owns full scope    ║
    ║   └──────┬──────┘                              ║
    ║     ┌────┼─────────┐                           ║
    ║     ▼    ▼         ▼                           ║
    ║   ┌────┐ ┌────┐ ┌────┐                        ║
    ║   │SUB1│ │SUB2│ │SUB3│  depth=1, narrow scope  ║
    ║   └──┬─┘ └──┬─┘ └────┘                        ║
    ║      │      │                                  ║
    ║      ▼      ▼                                  ║
    ║   ┌────┐ ┌────┐        depth=2, even narrower  ║
    ║   │SUB4│ │SUB5│                                ║
    ║   └────┘ └────┘                                ║
    ╚════════════════════════════════════════════════╝
            │ emit tasks
            ▼
    ┌─────────────────────────────────────────────┐
    │              TASK QUEUE (priority)           │
    │  [critical] [high] [normal] [low]           │
    └───────┬─────────────────────────────────────┘
            │ workers pick up tasks
            ▼
    ╔═══════╧════════════════════════════════════╗
    ║         WORKER POOL (parallel)             ║
    ║                                            ║
    ║  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ║
    ║  │ W-01 │  │ W-02 │  │ W-03 │  │ W-04 │  ║
    ║  │ own  │  │ own  │  │ own  │  │ own  │  ║
    ║  │ repo │  │ repo │  │ repo │  │ repo │  ║
    ║  │ copy │  │ copy │  │ copy │  │ copy │  ║
    ║  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  ║
    ╚═════╧═════════╧═════════╧═════════╧═══════╝
          │ structured handoffs flow UP
          ▼
    ┌─────────────────────────────────────────────┐
    │         HANDOFF ROUTING                     │
    │  worker handoff → parent planner inbox      │
    │  planner receives → reassesses → new tasks  │
    └─────────────────────────────────────────────┘
```

## Each Orchestration Cycle

```
┌─────────────────────────────────────────────────────────┐
│ CYCLE N                                                 │
│                                                         │
│  Phase 1: PLANNER PHASE                                 │
│    • Drain pending handoffs for each active planner     │
│    • Process handoffs → assess progress                 │
│    • Analyze codebase state → identify gaps             │
│    • Emit new tasks → push to priority queue            │
│    • Spawn subplanners if scope is large                │
│                                                         │
│  Phase 2: WORKER PHASE                                  │
│    • Pull batch of tasks from queue (up to max_workers) │
│    • Create isolated git worktree per worker            │
│    • Execute all tasks in parallel (ThreadPool)         │
│    • Each worker: read files → code → write → commit    │
│    • Collect handoffs, route to parent planners          │
│    • Retry failed tasks (up to max_retries)             │
│    • Cleanup worktrees                                  │
│                                                         │
│  Phase 3: FRESHNESS PHASE                               │
│    • For each planner:                                  │
│      - Rewrite scratchpad if interval reached           │
│      - Compress/summarize if context too large          │
│                                                         │
│  Phase 4: METRICS & COMPLETION CHECK                    │
│    • Log: commits/hr, tasks done, error rate            │
│    • If root planner reports complete → stop            │
│    • If all planners done + queue empty → stop          │
│                                                         │
│  → Sleep → Next cycle                                   │
└─────────────────────────────────────────────────────────┘
```

## Usage

```bash
# Install
pip install dspy

# Run
python multi_agent_system.py \
  --repo /path/to/your/project \
  --instructions "Build a REST API with user authentication, CRUD for blog posts, rate limiting, and comprehensive tests" \
  --model openai/gpt-4o \
  --workers 10 \
  --depth 3 \
  --iterations 50
```

## DSPy Signatures (LLM Decision Points)

| Signature | Input | Output | Used By |
|-----------|-------|--------|---------|
| `AnalyzeCodebase` | goal + codebase state + handoffs | analysis + priorities | Planner |
| `PlanTasks` | goal + analysis + scratchpad | tasks[] + subplanner scopes[] + scratchpad | Planner |
| `ReviewHandoffs` | goal + handoffs + state | assessment + follow_up + is_complete | Planner |
| `ExecuteTask` | task + code + file tree | code changes + tests + handoff report | Worker |
| `RewriteScratchpad` | old pad + new info + goal | fresh scratchpad | Freshness |
| `SummarizeContext` | long context + goal | concise summary | Freshness |

## Design Principles

1. **Anti-fragile**: Worker/planner failures are caught, logged, retried. System continues.
2. **Throughput-first**: Small error rate accepted. No 100% correctness gating.
3. **Information flows UP only**: Workers → Planner (via handoffs). No cross-talk.
4. **Isolation**: Each worker has own repo copy. No shared mutable state.
5. **Freshness**: Scratchpads rewritten (not appended). Context compressed when large.
6. **No central bottleneck**: No integrator gate. Workers commit independently.