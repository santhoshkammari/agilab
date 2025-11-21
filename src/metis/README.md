# Metis

Minimal async framework for LLM operations, agent execution, and evaluation.

## Quick Start

```python
import metis

lm = metis.LM()
```

Then use:
- `metis.LM` - Language model interface
- `metis.step()`, `metis.agent()` - Agent framework
- `metis.eval_batch()`, `metis.eval_stream()`, `metis.eval_example()` - Evaluation

---

## LM

Async LLM interface with streaming and batch support.

### API

```python
import metis

lm = metis.LM()

# Streaming
async for chunk in lm.stream(messages):
    print(chunk)

# Batch (parallel)
results = await lm.batch([messages1, messages2, messages3])
```

### Example

```python
import metis

lm = metis.LM()
messages = [{"role": "user", "content": "What is 2+2?"}]

async for chunk in lm.stream(messages):
    if chunk.get("choices"):
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
            print(delta["content"], end="")
```

---

## Agent

Async agent framework with streaming, tool calling, and multi-turn loops.

### Three Levels

1. **`metis.gen()`** - Low-level streaming (yields text chunks and tool calls)
2. **`metis.step()`** - Single LLM generation with async tool execution
3. **`metis.agent()`** - Multi-turn loop with max iterations

### API

```python
import metis

lm = metis.LM()

# Full agent loop
result = await metis.agent(lm=lm, history=history, tools=tools, max_iterations=5)

# Single step
result = await metis.step(lm=lm, history=history, tools=tools)
history.append(result.message)
if result.tool_calls:
    tool_results = await result.tool_results()
    history.extend([tr.message for tr in tool_results])

# Low-level streaming
async for chunk in metis.gen(lm=lm, history=history, tools=tools):
    if isinstance(chunk, metis.AssistantResponse):
        print(chunk.content, end="")
    elif isinstance(chunk, metis.ToolCall):
        print(f"[Tool] {chunk.name}")
```

### Data Structures

```python
metis.StepResult
    message: dict              # Assistant message
    tool_calls: list[dict]     # Tool calls made
    async def tool_results() -> list[metis.ToolResult]

metis.ToolResult
    tool_call_id: str
    output: str
    is_error: bool = False
    @property
    def message(self) -> dict  # Convert to history message format
```

### Tool Definition

```python
def my_tool(param1: str, param2: int) -> str:
    """Tool description.

    Args:
        param1: Description
        param2: Description
    """
    return "result"
```

---

## Eval

Async evaluation framework with streaming and batch support.

### Three Levels

1. **`metis.eval_example()`** - Evaluate single example
2. **`metis.eval_stream()`** - Stream results one-by-one
3. **`metis.eval_batch()`** - Batch evaluate with parallelization

### API

```python
import metis

lm = metis.LM()

# Single
result = await metis.eval_example(history, target, metric, lm=lm)

# Streaming
async for result in metis.eval_stream(histories, targets, metric, lm=lm):
    print(f"Score: {result.score}")

# Batch
result = await metis.eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    batch_size=4,
    parallel=True
)
print(f"Score: {result['score']:.1f}%")
```

### Metrics

```python
def exact_match(target, prediction):
    return 1.0 if prediction == target else 0.0

def contains_match(target, prediction):
    return 1.0 if target in prediction else 0.0
```

### With Tools

```python
result = await metis.eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    tools=[calculator],
    use_agent=True,  # Full agent loop
    max_iterations=5
)
```

---

## All Exports

```python
import metis

# LM
metis.LM

# Agent
metis.gen
metis.step
metis.agent
metis.AssistantResponse
metis.ToolCall
metis.StepResult
metis.ToolResult

# Eval
metis.eval_example
metis.eval_stream
metis.eval_batch
metis.EvalResult
```

## Sample Files

| File | Description |
|------|-------------|
| `sample_lm.py` | LM streaming and batch demos |
| `sample_agent.py` | Agent step() and agent() demos |
| `sample_eval.py` | Evaluation framework demos |

---

## Design

- Minimal - Functions over classes
- Async-first - Full async/await support
- Composable - Each level builds on previous
- Observable - Streaming for real-time output
- Production-ready - Error handling, progress bars, parallelization
