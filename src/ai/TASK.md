# Parallel Agents — Cursor-Style Self-Driving Codebase

## What We Built

A recursive parallel planning agent system inspired by [Cursor's blog post](https://cursor.com/blog/self-driving-codebases) on self-driving codebases.

### Files

| File | Purpose |
|------|---------|
| `ai.py` | Core library — LM, Predict, Prediction, Module, Eval. DSPy-style API with sync/async, streaming, tool calling, verbose logging |
| `file_tools.py` | Plain function wrappers around langchain-community file tools + custom `edit_file`. 7 tools: read, write, edit, list, copy, move, delete |
| `plan_agent.py` | Recursive parallel planner + isolated workers. The main agent harness |
| `cursor.py` | Quick test script for ai.py imports and basic Predict usage |
| `parallel.txt` | Original architecture vision doc |

---

## Architecture (Final Design)

```
Planner (depth=0, root)
  ├── SPLIT → spawns SubPlanners in parallel
  │     ├── Planner(d=1) → WORKER → Worker (isolated, tools only)
  │     └── Planner(d=1) → WORKER → Worker (isolated, tools only)
  ├── Receives handoffs from all children
  ├── Loops ONLY if real concerns exist
  └── Writes scratchpad.md each iteration (rewrite, never append)
```

### Strict Role Separation

- **Planner**: Pure LLM, zero tools. Only plans and delegates. Never codes.
- **Worker**: Tools only, no planning awareness. Executes one focused task. Unaware of other agents or the larger system.
- **Git**: Shared truth layer. Workers commit after changing files.
- **Scratchpad**: Root planner's working memory, rewritten each loop.

### Key Classes

- `Planner` — recursive, spawns subplanners or workers. Only root (depth=0) loops.
- `Worker` — executes with file_tools, returns structured `Handoff`.
- `Handoff` — dataclass with `what_done`, `findings`, `concerns`, `files_changed`. Parsed from worker JSON via `Handoff.from_json()`.

---

## Evolution & Lessons Learned

### Step 1: Basic ai.py test
- `import ai` → `ai.LM()` → `ai.configure(lm)` → `ai.Predict("q -> answer")` works
- Tool calling works: `Predict("q -> answer", tools=[get_weather])` calls tools via vLLM
- `verbose=True` on Predict gives `[tool]`, `[result]`, `[done]` logging to stderr

### Step 2: file_tools.py
- langchain-community has Read, Write, List, Copy, Move, Delete — but as BaseTool classes
- Wrapped them as plain Python functions with typed args + docstrings (required by `get_json_schema`)
- Added custom `edit_file()` (exact string replace) — langchain doesn't have one
- **Gotcha**: `transformers.get_json_schema()` requires docstring descriptions for ALL args or it throws `DocstringParsingException`

### Step 3: First PlanAgent (combined planner+worker)
- Single class that decides SPLIT or EXECUTE
- Worked but violated Cursor's key insight: **too many roles = pathological behavior**
- Model would hallucinate results instead of calling tools when given multi-field output signatures

### Step 4: Strict Planner/Worker split
- Planner has zero tools, Worker has no planning awareness
- Worker uses single output field `"task -> handoff"` to avoid hallucination
- Multi-field signatures (`task -> what_done, findings, concerns`) caused model to fill all fields without using tools

### Step 5: Rich Handoff parsing
- Workers naturally return JSON in their handoff string
- `Handoff.from_json()` extracts structured fields, falls back to raw string
- Removed `suggestions` field — models use it to invent busywork

### Step 6: Planner continuous loop
- Root planner receives handoffs, checks for concerns, decides if more work needed
- **Critical bug found**: loop ran even with no concerns, wasting LLM calls
- **Fix**: no concerns = instant break BEFORE any LLM call. Only consult loop LLM when real concerns exist.

### Step 7: Subplanners must NOT loop
- Cursor's design: subplanners are fire-and-forget. Execute once, return handoff up.
- Only root planner loops. `if self.depth == 0:` guard on the loop block.
- Previous version had all planners looping → subplanners invented extra work ("analyze code structure", "verify accuracy") that nobody asked for.

### Step 8: Git commit + Scratchpad
- Workers auto-commit via `git add + commit` when `files_changed` is non-empty
- Root planner writes `scratchpad.md` each loop iteration (rewrite, never append)
- **Critical bug**: stale scratchpad from previous runs leaked into new runs, making planner think task was already done
- **Fix**: `run_plan()` deletes scratchpad at start of fresh run

---

## Pathologies Encountered (matching Cursor's findings)

| Pathology | Cursor's Version | Our Version | Fix |
|-----------|-----------------|-------------|-----|
| Agent does everything | Continuous executor overwhelmed | PlanAgent both planned and coded | Strict Planner/Worker split |
| Workers go off-scope | "Workers would fix irrelevant things" | Worker analyzed code nobody asked for | Tighter prompt: "ONLY the task, nothing more" |
| Loop invents busywork | "Claimed premature completion" / "Refused to plan" | Subplanners looped and spawned extra workers | Only root loops, subplanners are single-shot |
| Hallucinated results | N/A (they used real tools) | Multi-field signature → model filled fields without calling tools | Single `handoff` output field |
| Stale context | "Scratchpad should be rewritten not appended" | Old scratchpad made planner skip work | Clear scratchpad on fresh run |
| Suggestions as work | N/A | Workers suggested improvements, planners treated them as tasks | Removed suggestions field, loop prompt ignores nice-to-haves |

---

## Prompt Engineering Lessons

1. **Constraints > Instructions** — "Do NOT analyze beyond scope" works better than "Focus on the task"
2. **"Do X and Y" = ALWAYS SPLIT** — had to explicitly say this or model combines into single worker
3. **Depth bias** — "At depth 0-1 aggressively split, at depth 2+ prefer WORKER"
4. **Concrete numbers** — "3-8 subtasks" produces better results than "multiple subtasks"
5. **Default to DONE** — loop prompt must say "Default to DONE. Suggestions are NOT reasons to continue."
6. **Single output field for tool-using agents** — multi-field signatures cause hallucination over tool use

---

## Performance

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| LLM calls for simple 2-part task | 12+ | 5 |
| Wasted loop iterations | 4-6 | 0 |
| Workers spawned off-scope | 2-3 | 0 |
| Tool hallucination | Frequent | None |

---

### Step 9: Sequential dependency detection
- Task "websearch then summarize" was split in parallel — summarizer ran with zero input
- Root cause: prompt said "do X and Y = ALWAYS SPLIT" without checking dependencies
- **Fix**: prompt now asks "does step B need step A's OUTPUT to begin? If yes → WORKER"
- Key insight: a worker CAN do multiple sequential steps. SPLIT is only for truly independent work.

### Step 10: Connection timeout under load
- 15+ agents hitting LLM server simultaneously → `ConnectionTimeoutError` after 10s
- Default `aiohttp.ClientTimeout(connect=10)` too tight for concurrent agents
- **Fix**: `connect=120` (2 minutes) in plan_agent.py AI Config

### Step 11: PyAutoGUI search — jitter + serialization
- `web_gui.py` search tool uses PyAutoGUI to open browser, copy HTML, parse results
- Problem: fixed `time.sleep()` values + no concurrency control = Google bot detection + race conditions
- **Fixes applied to `mcp_tools/web_gui.py`**:
  1. `_search_lock` (threading.Lock) — serializes searches since pyautogui controls one mouse/keyboard
  2. `_jitter(low, high)` — random delay replacing all fixed sleeps
  3. Timing: page load `2.5-5.0s`, between actions `0.3-1.5s`, cooldown `0.5-1.0s`
- Lock is critical: even if 10 agents call `search()` simultaneously, they queue and run one at a time

---

## What's Next

1. **Git worktrees** — each worker gets isolated repo copy (Cursor's approach)
2. **Bigger test** — multi-file coding task to test real parallel execution
3. **Context passing** — planner passes relevant codebase context down to workers
4. **Error recovery** — worker fails → handoff with concern → root replans
5. **Token budget** — track total tokens spent across all agents
