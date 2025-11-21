# Evaluation Framework

Minimal, async evaluation system for agilab. Mirrors LM interface (`stream`/`batch` pattern) for consistency.

## Overview

Three-level evaluation API:

1. **`eval_example()`** - Evaluate single example
2. **`eval_stream()`** - Stream results one-by-one (real-time monitoring)
3. **`eval_batch()`** - Batch evaluate with optional parallelization

## Core Design

Evaluates conversation histories on a dataset using custom metrics:

```python
from lm import LM

# Define your metric
def exact_match(target, prediction):
    return 1.0 if prediction == target else 0.0

# Dataset with histories and targets
data = [
    {
        "history": [{"role": "user", "content": "What is 2+2?"}],
        "target": "4"
    }
]

histories = [d["history"] for d in data]
targets = [d["target"] for d in data]

lm = LM()

# Single evaluation
result = await eval_example(histories[0], targets[0], exact_match, lm=lm)
print(result)  # EvalResult(score=1.0, prediction="4", ...)

# Streaming (real-time)
async for result in eval_stream(histories, targets, exact_match, lm=lm):
    print(f"Running avg: {result.score}")

# Batch (sequential or parallel)
result = await eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    batch_size=4,
    parallel=True,
    progress=True
)
print(f"Final: {result['score']:.1f}% ({result['passed']}/{result['total']})")
```

## API Reference

### `eval_example(history, target, metric, lm, tools=None, use_agent=False, **kwargs)`

Evaluate a single conversation history.

**Args:**
- `history`: List of message dicts (conversation history)
- `target`: Expected answer/output
- `metric`: Callable(target, prediction) → float [0-1]
- `lm`: Language model instance
- `tools`: Optional list of tool functions
- `use_agent`: If `True`, use full agent loop; else single step
- `**kwargs`: Extra params (e.g., `max_iterations`)

**Returns:**
```python
EvalResult(
    history=history,
    prediction=pred,
    score=float,
    metadata=dict
)
```

---

### `eval_stream(histories, targets, metric, lm, tools=None, use_agent=False, **kwargs)`

Stream evaluation results one-by-one.

**Args:**
- `histories`: List of conversation histories
- `targets`: List of expected answers
- `metric`: Callable(target, prediction) → float
- `lm`: Language model instance
- `tools`: Optional list of tool functions
- `use_agent`: If `True`, use full agent loop; else single step
- `**kwargs`: Extra params

**Yields:** `EvalResult` for each example

**Usage:**
```python
async for result in eval_stream(histories, targets, metric, lm=lm):
    print(f"Score: {result.score}")
    if result.score < 0.5:
        break  # Early stopping
```

---

### `eval_batch(histories, targets, metric, lm, tools=None, use_agent=False, batch_size=4, parallel=False, progress=True, **kwargs)`

Batch evaluate conversation histories.

**Args:**
- `histories`: List of conversation histories
- `targets`: List of expected answers
- `metric`: Callable(target, prediction) → float
- `lm`: Language model instance
- `tools`: Optional list of tool functions
- `use_agent`: If `True`, use full agent loop; else single step
- `batch_size`: Size for concurrent batches (if `parallel=True`)
- `parallel`: If `True`, evaluate batches concurrently
- `progress`: Show progress bar
- `**kwargs`: Extra params (e.g., `max_iterations`)

**Returns:**
```python
{
    "score": float,        # Percentage (0-100)
    "passed": int,         # Count where score > 0.5
    "total": int,          # Total examples
    "results": List[EvalResult]
}
```

---

## Examples

### Example 1: Simple QA Evaluation

```python
from eval import eval_batch
from lm import LM

# Module - uses history-based evaluation
async def qa_module(history: list) -> str:
    lm = LM()
    response = await lm.stream(history)
    return response

# Metric
def exact_match(target, prediction):
    return 1.0 if prediction == target else 0.0

# Dataset
data = [
    {
        "history": [{"role": "user", "content": "Q: What is 2+2?"}],
        "target": "4"
    },
    {
        "history": [{"role": "user", "content": "Q: What is 5+5?"}],
        "target": "10"
    },
]

histories = [d["history"] for d in data]
targets = [d["target"] for d in data]

# Evaluate
result = await eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=LM()
)

print(result)  # {'score': 85.0, 'passed': 2, 'total': 2, ...}
```

### Example 2: Multiple Metrics

```python
def exact_match(target, prediction):
    return 1.0 if prediction == target else 0.0

def contains_match(target, prediction):
    return 1.0 if target in prediction else 0.0

# Compare metrics
result1 = await eval_batch(histories, targets, exact_match, lm=LM())
result2 = await eval_batch(histories, targets, contains_match, lm=LM())

print(f"Exact: {result1['score']:.1f}%")
print(f"Contains: {result2['score']:.1f}%")
```

### Example 3: Custom Postprocessing

```python
from eval import eval_example

# Process prediction before scoring
def extract_number(prediction):
    import re
    match = re.search(r'\d+', prediction)
    return match.group(0) if match else "unknown"

history = [{"role": "user", "content": "What is 2+2?"}]
target = "4"

result = await eval_example(history, target, exact_match, lm=LM())
# Postprocess if needed
result.prediction = extract_number(result.prediction)
```

---

## Data Structures

### EvalResult

```python
@dataclass
class EvalResult:
    history: list          # Original conversation history
    prediction: Any        # Module's prediction
    score: float           # Metric score [0-1]
    metadata: dict         # Additional info (errors, etc)
```

---

## Evaluation Modes

### Mode 1: History-Based Evaluation (Default)

Evaluate using conversation histories with `agent.step()`:

```python
from lm import LM

lm = LM()

# Dataset with histories and targets
data = [
    {
        "history": [{"role": "user", "content": "What is 2+2?"}],
        "target": "4"
    }
]

histories = [d["history"] for d in data]
targets = [d["target"] for d in data]

# Evaluate
result = await eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    use_agent=False  # Single step mode
)
```

### Mode 2: Agent-Based Evaluation with Tools

Evaluate using `agent.step()` or full agent loop with tool support:

```python
from lm import LM

# Tools
def calculator(expr: str) -> str:
    return str(eval(expr))

lm = LM()
tools = [calculator]

# Single step evaluation
result = await eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    tools=tools,
    use_agent=False  # Single step with tools
)

# Or full agent loop for multi-step tasks
result = await eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    tools=tools,
    use_agent=True,  # Full agent loop
    max_iterations=5
)
```

The framework will:
1. Use conversation history directly
2. Call `agent.step(lm, history, tools)` or `agent.agent(lm, history, tools)`
3. Execute tool calls if needed
4. Collect results into prediction
5. Apply metric

---

## Design Principles

1. **Minimal** - No classes, just functions
2. **Async-first** - Full async/await support
3. **Flexible** - Works with any callable module and metric
4. **Observable** - Streaming for real-time monitoring
5. **Production-ready** - Error handling, progress bars, parallelization
6. **Dual-mode** - Direct evaluation OR agent-based with tools

---

## See Also

- `eval_run_sample.py` - Working examples with real LM (includes basic and tool-based demos)
- `agent.py` - Agent framework with step() and agent() for multi-turn conversations
- `lm.py` - Language model interface
