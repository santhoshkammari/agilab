# Agent Framework

Minimal async agent framework with streaming, tool calling, and multi-turn loops.

## Overview

Three levels of abstraction:

1. **`gen()`** - Low-level streaming (yields text chunks and tool calls)
2. **`step()`** - Single LLM generation with async tool execution
3. **`agent()`** - Multi-turn loop with max iterations

## Core Functions

### `gen(lm, history, tools=None)`

Stream LLM response chunks and tool calls.

**Yields:**
- `AssistantResponse(content)` - Text chunks
- `ToolCall(id, name, arguments)` - Tool calls

**Usage:**
```python
async for chunk in gen(lm=lm, history=history, tools=tools):
    if isinstance(chunk, AssistantResponse):
        print(chunk.content, end="", flush=True)
    elif isinstance(chunk, ToolCall):
        print(f"[Tool] {chunk.name}")
```

---

### `step(lm, history, tools=None, early_tool_execution=True)`

Execute ONE LLM generation with async tool execution.

**Features:**
- Streams LLM response
- Spawns async tasks for tool calls (tools run in parallel)
- Returns immediately with tool futures still running
- Call `await result.tool_results()` to wait for outputs

**Args:**
- `lm`: Language model instance
- `history`: Conversation history (list of messages)
- `tools`: List of callable tools with docstrings
- `early_tool_execution`: Execute tools while LLM is still streaming

**Returns:**
```python
StepResult(
    message={...},           # Assistant message
    tool_calls=[...],        # List of tool calls
    usage=None,              # Token usage if available
    _tool_futures={...}      # Futures for tool results
)

# Wait for results
results = await step_result.tool_results()
```

**Usage:**
```python
result = await step(lm=lm, history=history, tools=tools)
history.append(result.message)

if result.tool_calls:
    tool_results = await result.tool_results()
    for tr in tool_results:
        history.append(tool_result_to_message(tr))
```

---

### `agent(lm, initial_message, tools=None, max_iterations=10, early_tool_execution=True)`

Execute a complete multi-turn agent loop.

**Features:**
- Automatically loops with `step()` until no more tool calls
- Handles tool result collection
- Stops at max_iterations
- Returns full history and statistics

**Args:**
- `lm`: Language model instance
- `initial_message`: User's starting message
- `tools`: List of callable tools
- `max_iterations`: Maximum steps to execute
- `early_tool_execution`: Execute tools while streaming

**Returns:**
```python
{
    "history": [...],        # Full conversation
    "iterations": int,       # Number of steps
    "final_response": str,   # Last assistant message
    "tool_calls_total": int  # Total tool calls made
}
```

**Usage:**
```python
result = await agent(
    lm=lm,
    initial_message="Solve this problem and verify the answer",
    tools=[calculator, verifier],
    max_iterations=5
)

print(f"Solved in {result['iterations']} steps")
print(result['final_response'])
```

---

## Data Structures

### AssistantResponse
```python
@dataclass
class AssistantResponse:
    content: str  # Text chunk from LLM
```

### ToolCall
```python
@dataclass
class ToolCall:
    id: str          # Unique tool call ID
    name: str        # Tool function name
    arguments: str   # JSON arguments
```

### StepResult
```python
@dataclass
class StepResult:
    message: dict                        # Assistant message
    tool_calls: list[dict]              # Tool calls made
    usage: dict | None                  # Token usage
    _tool_futures: dict[str, Future]    # Tool result futures

    async def tool_results() -> list[ToolResult]:
        """Wait for all tool executions and return results"""
```

### ToolResult
```python
@dataclass
class ToolResult:
    tool_call_id: str  # ID from tool call
    output: str        # Result/output
    is_error: bool     # True if execution failed
```

---

## Usage Patterns

### Pattern 1: Simple Agent Loop
```python
from agent import agent
from lm import LM

def calculator(expr: str) -> str:
    """Calculate math expression"""
    return str(eval(expr))

lm = LM()
result = await agent(
    lm=lm,
    initial_message="What is 2+2? Then multiply by 5.",
    tools=[calculator],
    max_iterations=3
)

print(result['final_response'])
```

### Pattern 2: Manual Step-by-Step Loop
```python
from agent import step, tool_result_to_message
from lm import LM

lm = LM()
history = [{"role": "user", "content": "question"}]
tools = [tool1, tool2]

for i in range(max_iterations):
    result = await step(lm=lm, history=history, tools=tools)
    history.append(result.message)

    if result.tool_calls:
        tool_results = await result.tool_results()
        for tr in tool_results:
            history.append(tool_result_to_message(tr))
    else:
        break  # Done
```

### Pattern 3: Real-Time Streaming
```python
from agent import gen, ToolCall, AssistantResponse
from lm import LM

lm = LM()
messages = [{"role": "user", "content": "question"}]

async for chunk in gen(lm=lm, history=messages, tools=tools):
    if isinstance(chunk, AssistantResponse):
        print(chunk.content, end="", flush=True)
    elif isinstance(chunk, ToolCall):
        print(f"\n[Calling {chunk.name}]")
```

---

## Key Features

✅ **Async/Parallel Tool Execution**
- Tools run in parallel, not sequential
- Early execution while LLM is still streaming
- Optional early execution disable

✅ **Minimal API**
- Just functions, no classes
- Clear separation of concerns (gen → step → agent)
- Works with any tool (sync or async)

✅ **Observable**
- Stream LLM responses in real-time
- Get tool calls as they're generated
- Full conversation history

✅ **Production-Ready**
- Error handling in tool execution
- Proper message formatting
- Token usage tracking (ready for integration)

---

## Tool Requirements

Tools must be regular Python functions with:
- Clear parameter types and docstrings
- Can be sync or async
- Return string or object convertible to string

```python
def my_tool(param1: str, param2: int) -> str:
    """
    Tool description and parameter docs.

    Args:
        param1: Description
        param2: Description
    """
    return "result"

async def my_async_tool(param: str) -> str:
    """Async tool description"""
    return "result"
```

---

## Examples

See sample files:
- `agent_run_sample.py` - Low-level gen() and step() examples
- `agent_loop_sample.py` - High-level agent() loop examples

---

## Design Philosophy

**Three-level abstraction:**
1. `gen()` - Handles streaming and message parsing
2. `step()` - Adds tool execution and async handling
3. `agent()` - Adds looping and iteration management

Each level is independently useful but designed to compose together.
