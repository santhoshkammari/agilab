# Agilab AI Framework

Asynchronous framework for LLM operations, agent execution, and evaluation with streaming capabilities.

## Overview

This framework provides a minimal, async-first approach to working with Large Language Models, featuring:
- Streaming and batch LLM operations
- Agent framework with tool calling support
- Comprehensive evaluation utilities
- Support for multi-turn conversations and complex workflows

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai
```

## Quick Start

```python
import ai

# Initialize the language model
lm = ai.LM(api_base="http://localhost:8000", model="your-model-name")

# Configure globally (optional)
ai.configure(lm=lm)

# Use the framework
async def main():
    # Define tools
    def calculator(expression: str) -> str:
        """Simple calculator for math expressions"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # Create conversation history
    history = [{"role": "user", "content": "What is 2+2?"}]

    # Run agent with tools
    async for result in ai.agent(lm=lm, history=history, tools=[calculator]):
        print(result)

# Run the async function
import asyncio
asyncio.run(main())
```

## Components

### LM (Language Model Interface)

Async LLM interface with streaming and batch support.

#### API

```python
import ai

lm = ai.LM(
    model="your-model",           # Model identifier
    api_base="http://localhost:8000",  # API endpoint
    api_key="your-api-key"        # Authentication
)

# Streaming
async for chunk in lm.stream(messages):
    print(chunk)

# Batch (parallel)
results = await lm.batch([messages1, messages2, messages3])
```

#### Example

```python
import ai
import asyncio

async def demo_streaming():
    lm = ai.LM(api_base="http://localhost:8000")
    await lm.start()  # Initialize HTTP session

    messages = [{"role": "user", "content": "What is 2+2?"}]

    async for chunk in lm.stream(messages):
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                print(delta["content"], end="")

    await lm.close()  # Clean up HTTP session

asyncio.run(demo_streaming())
```

---

### Agent Framework

Async agent framework with streaming, tool calling, and multi-turn loops.

#### Three Levels of Abstraction

1. **`ai.gen()`** - Low-level streaming (yields text chunks and tool calls)
2. **`ai.step()`** - Single LLM generation with async tool execution
3. **`ai.agent()`** - Multi-turn loop with max iterations

#### API

```python
import ai

lm = ai.LM(api_base="http://localhost:8000")

# Full agent loop
async for result in ai.agent(lm=lm, history=history, tools=tools, max_iterations=5):
    print(result)

# Single step
async for result in ai.step(lm=lm, history=history, tools=tools):
    print(result)
if hasattr(result, 'tool_results'):
    tool_results = await result.tool_results()
    print(tool_results)

# Low-level streaming
async for chunk in ai.gen(lm=lm, history=history, tools=tools):
    if isinstance(chunk, ai.AssistantResponse):
        print(chunk.content, end="")
    elif isinstance(chunk, ai.ToolCall):
        print(f"[Tool] {chunk.name}: {chunk.arguments}")
```

#### Data Structures

```python
ai.StepResult
    message: dict              # Assistant message
    tool_calls: list[dict]     # Tool calls made
    usage: dict | None         # Token usage info
    async def tool_results() -> list[ai.ToolResult]

ai.ToolResult
    tool_call_id: str
    output: str
    is_error: bool = False
    @property
    def message(self) -> dict  # Convert to history message format

ai.AgentResult
    history: list[dict]        # Full conversation history
    iterations: int            # Number of iterations performed
    final_response: str        # Final response from the agent
    tool_calls_total: int      # Total number of tool calls made
```

#### Tool Definition

```python
def my_tool(param1: str, param2: int = 10) -> str:
    """
    Tool description for what this function does.

    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter (optional, default 10)
    """
    # Implementation here
    return "result"
```

#### Complete Agent Example

```python
import asyncio
from ai.agent import LM, agent

def get_weather(city: str, unit: str = "celsius"):
    """Get the current weather for a city"""
    return f"Weather in {city}: 22 degrees {unit}"

async def run_agent():
    lm = LM(api_base="http://localhost:8000")
    await lm.start()

    tools = [get_weather]
    history = [{"role": "user", "content": "What is the weather in London?"}]

    async for result in agent(lm=lm, history=history, tools=tools, max_iterations=5):
        print(result)

    await lm.close()

asyncio.run(run_agent())
```

---

### Evaluation Framework

Async evaluation framework with streaming and batch support for testing and validating your LLM applications.

#### Three Levels of Evaluation

1. **`ai.eval_example()`** - Evaluate single example
2. **`ai.eval_stream()`** - Stream results one-by-one (for real-time monitoring)
3. **`ai.eval_batch()`** - Batch evaluate with parallelization options

#### API

```python
import ai

lm = ai.LM(api_base="http://localhost:8000")

# Single example evaluation
result = await ai.eval_example(history, target, metric, lm=lm)

# Streaming evaluation (useful for monitoring)
async for result in ai.eval_stream(histories, targets, metric, lm=lm):
    print(f"Score: {result.score}")

# Batch evaluation with parallelization
result = await ai.eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    batch_size=4,
    parallel=True
)
print(f"Average Score: {result['score']:.1f}%")
```

#### Common Metrics

```python
def exact_match(target, prediction):
    """Exact string match (case-sensitive)"""
    return 1.0 if str(target).strip() == str(prediction).strip() else 0.0

def contains_match(target, prediction):
    """Check if prediction contains the target"""
    return 1.0 if str(target).lower() in str(prediction).lower() else 0.0

def fuzzy_match(target, prediction, threshold=0.8):
    """Fuzzy string matching using similarity"""
    # Implementation would use a library like difflib or fuzzywuzzy
    # This is a conceptual example
    pass
```

#### Evaluation with Tools

```python
# Evaluate using single step with tools
result = await ai.eval_example(
    history=[{"role": "user", "content": "What is 2+2?"}],
    target="4",
    metric=exact_match,
    lm=lm,
    tools=[calculator],  # Using tools during evaluation
    use_agent=False
)

# Evaluate using full agent loop with tools
result = await ai.eval_batch(
    histories=histories,
    targets=targets,
    metric=exact_match,
    lm=lm,
    tools=[calculator, search],
    use_agent=True,      # Use full agent loop
    max_iterations=5
)
```

#### Complete Evaluation Example

```python
import asyncio
from ai.eval import eval_batch
from ai.agent import LM

def exact_match(target, prediction):
    return 1.0 if str(target).strip() == str(prediction).strip() else 0.0

async def run_evaluation():
    lm = LM(api_base="http://localhost:8000")
    await lm.start()

    # Sample dataset
    histories = [[{"role": "user", "content": "What is 2+2?"}]]
    targets = ["4"]

    # Run evaluation
    result = await eval_batch(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        parallel=False
    )

    print(f"Evaluation Score: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")

    await lm.close()

asyncio.run(run_evaluation())
```

---

## All Exports

```python
import ai

# LM Interface
ai.LM
ai.configure      # Configure global LM
ai.get_lm         # Get current LM from context

# Agent Framework
ai.gen            # Low-level streaming
ai.step           # Single generation with tools
ai.agent          # Multi-turn agent loop
ai.AssistantResponse
ai.ToolCall
ai.StepResult
ai.ToolResult
ai.AgentResult

# Evaluation Framework
ai.eval_example   # Single example evaluation
ai.eval_stream    # Streaming evaluation
ai.eval_batch     # Batch evaluation
ai.EvalResult     # Evaluation result data structure
```

## Sample Files

| File | Description |
|------|-------------|
| `sample_lm.py` | Demonstrates LM streaming and batch operations |
| `sample_agent.py` | Shows agent step() and agent() usage with tools |
| `sample_eval.py` | Comprehensive evaluation framework demonstrations |

---

## Architecture

- **Minimal Design**: Functions over classes, keeping the API simple
- **Async-First**: Full async/await support for non-blocking operations
- **Composable**: Each level builds on the previous, allowing flexible usage
- **Observable**: Streaming interfaces for real-time output and monitoring
- **Production-Ready**: Includes error handling, progress tracking, and parallelization

## Setup for Development

```bash
# Install dependencies (if any)
pip install aiohttp transformers

# Run samples to test
python -m ai.sample_lm
python -m ai.sample_agent
python -m ai.sample_eval
```

## Contributing

This framework is designed to be lightweight and focused. Contributions should maintain the minimal philosophy while extending functionality in a consistent way.
