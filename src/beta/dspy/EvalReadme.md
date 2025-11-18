# Evaluation Framework

Minimal, async evaluation system for agilab. Mirrors LM interface (`stream`/`batch` pattern) for consistency.

## Overview

Three-level evaluation API:

1. **`eval_example()`** - Evaluate single example
2. **`eval_stream()`** - Stream results one-by-one (real-time monitoring)
3. **`eval_batch()`** - Batch evaluate with optional parallelization

## Core Design

Evaluates a module on a dataset using custom metrics:

```python
# Define your module
async def my_module(question: str) -> str:
    response = await lm.stream(...)
    return response

# Define your metric
def exact_match(example, prediction):
    return 1.0 if prediction == example.answer else 0.0

# Single evaluation
result = await eval_example(my_module, example, exact_match)
print(result)  # EvalResult(score=1.0, prediction="4", ...)

# Streaming (real-time)
async for result in eval_stream(my_module, examples, exact_match):
    print(f"Running avg: {result.score}")

# Batch (sequential or parallel)
result = await eval_batch(
    module_fn=my_module,
    examples=dev_set,
    metric=exact_match,
    batch_size=4,
    parallel=True,
    progress=True
)
print(f"Final: {result['score']:.1f}% ({result['passed']}/{result['total']})")
```

## API Reference

### `eval_example(module_fn, example, metric, **kwargs)`

Evaluate a single example.

**Args:**
- `module_fn`: Callable that takes `**inputs` and returns prediction
- `example`: Example object with `.inputs()` method
- `metric`: Callable(example, prediction) → float [0-1]
- `**kwargs`: Extra params passed to module_fn

**Returns:**
```python
EvalResult(
    example=example,
    prediction=pred,
    score=float,
    metadata=dict
)
```

---

### `eval_stream(module_fn, examples, metric, **kwargs)`

Stream evaluation results one-by-one.

**Args:**
- `module_fn`: Callable that takes `**inputs` and returns prediction
- `examples`: List of examples
- `metric`: Callable(example, prediction) → float
- `**kwargs`: Extra params passed to module_fn

**Yields:** `EvalResult` for each example

**Usage:**
```python
async for result in eval_stream(module, examples, metric):
    print(f"Score: {result.score}")
    if result.score < 0.5:
        break  # Early stopping
```

---

### `eval_batch(module_fn, examples, metric, batch_size=4, parallel=False, progress=True, **kwargs)`

Batch evaluate examples.

**Args:**
- `module_fn`: Callable that takes `**inputs` and returns prediction
- `examples`: List of examples
- `metric`: Callable(example, prediction) → float
- `batch_size`: Size for concurrent batches (if `parallel=True`)
- `parallel`: If `True`, evaluate batches concurrently
- `progress`: Show progress bar
- `**kwargs`: Extra params passed to module_fn

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
from example import Example
from lm import LM

# Module
async def qa_module(question: str) -> str:
    lm = LM()
    messages = [{"role": "user", "content": f"Q: {question}"}]
    response = await lm.stream(messages)
    return response

# Metric
def exact_match(example, prediction):
    return 1.0 if prediction == example.answer else 0.0

# Dataset
examples = [
    Example(question="2+2?", answer="4"),
    Example(question="5+5?", answer="10"),
]

# Evaluate
result = await eval_batch(
    module_fn=qa_module,
    examples=examples,
    metric=exact_match
)

print(result)  # {'score': 85.0, 'passed': 2, 'total': 2, ...}
```

### Example 2: Multiple Metrics

```python
def exact_match(example, prediction):
    return 1.0 if prediction == example.answer else 0.0

def contains_match(example, prediction):
    return 1.0 if example.answer in prediction else 0.0

# Compare metrics
result1 = await eval_batch(module, examples, exact_match)
result2 = await eval_batch(module, examples, contains_match)

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

result = await eval_example(module, example, metric)
# Postprocess if needed
result.prediction = extract_number(result.prediction)
```

---

## Data Structures

### EvalResult

```python
@dataclass
class EvalResult:
    example: Any           # Original example
    prediction: Any        # Module's prediction
    score: float           # Metric score [0-1]
    metadata: dict         # Additional info (errors, etc)
```

---

## Design Principles

1. **Minimal** - No classes, just functions
2. **Async-first** - Full async/await support
3. **Flexible** - Works with any callable module and metric
4. **Observable** - Streaming for real-time monitoring
5. **Production-ready** - Error handling, progress bars, parallelization

---

## See Also

- `eval_run_sample.py` - Working examples with real LM
- `agent.py` - Agent framework for agentic evaluation
- `example.py` - Example data structure
- `lm.py` - Language model interface
