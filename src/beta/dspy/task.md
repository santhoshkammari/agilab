# dspy - Tasks for Tomorrow

## Overview
dspy is copied from aspy and ready for development. Basic structure is in place.

## Completed Today ✅
- [x] Copied aspy folder to dspy
- [x] Updated README from aspy to dspy
- [x] Created test_dspy.py test script
- [x] Verified imports work correctly

## Tasks for Tomorrow 🚀

### 1. Start vLLM Server
```bash
vllm serve /home/ng6309/datascience/santhosh/models/Qwen3-14B \
  --gpu-memory-utilization 0.8 \
  --max-model-len 60000 \
  --rope-scaling '{"type": "linear", "factor": 2.0}' \
  --rope-theta 1000000 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Test dspy Module
- Run `python test_dspy.py` to verify all components work:
  - dspy.LM (Language Model interface)
  - dspy.Predict (Signature-based prediction)
  - dspy.Agent (Streaming agent)

### 3. Integrate UniversalLogger
Following the FRAMEWORK_ANALYSIS.md recommendations:

#### Update LM class (`dspy/lm/lm.py`)
```python
from logger import UniversalLogger

class LM:
    def __init__(self, model="", api_base="http://localhost:8000", logger=None):
        self.log = logger or UniversalLogger("lm", subdir="llm_calls")

        # Log all LLM requests
        self.log.dev(f"LM initialized: {model}")

    async def _single_async(self, messages, **params):
        # Log request
        self.log.dev({
            "action": "llm_request",
            "model": self.model,
            "messages_count": len(messages)
        })

        response = await ...

        # Log response
        self.log.prod({
            "action": "llm_response",
            "prompt_tokens": response['usage']['prompt_tokens'],
            "completion_tokens": response['usage']['completion_tokens']
        })
```

#### Update Agent class (`dspy/agent.py`)
```python
from logger import UniversalLogger

class Agent:
    def __init__(self, system_prompt="", lm=None, tools=None, logger=None):
        self.log = logger or UniversalLogger("agent", subdir="agent_logs")

        # Log conversations
        self.log.ai(user_input, role="user")
        self.log.ai(response, role="assistant")
```

#### Update Predict class (`dspy/predict/predict.py`)
```python
from logger import UniversalLogger

class Predict:
    def __init__(self, signature, lm=None, logger=None):
        self.log = logger or UniversalLogger("predict", subdir="predictions")
```

### 4. Add Error Handling & Retry Logic
Add to LM class:
```python
async def _single_async(self, messages, **params):
    try:
        # Make request
        ...
    except asyncio.TimeoutError:
        self.log.error({"error": "LLM timeout"})
        return self._retry_with_backoff(messages, params)
    except aiohttp.ClientError as e:
        self.log.error({"error": "Connection failed", "details": str(e)})
        raise
```

### 5. Add Tool System
Create `dspy/tools/` module:
- `tools/registry.py` - Tool registration
- `tools/executor.py` - Tool execution with logging
- `tools/builtin.py` - Built-in tools (calculator, web_search, etc.)

Example:
```python
# dspy/tools/registry.py
from logger import UniversalLogger

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.log = UniversalLogger("tools", subdir="tool_calls")

    def register(self, func):
        self.tools[func.__name__] = func
        self.log.info(f"Registered tool: {func.__name__}")
        return func
```

### 6. Production Features

#### Cost Tracking
Add token cost calculation:
```python
# dspy/lm/lm.py
COST_PER_1K_TOKENS = {
    "input": 0.01,   # $0.01 per 1K input tokens
    "output": 0.03   # $0.03 per 1K output tokens
}

def _calculate_cost(self, prompt_tokens, completion_tokens):
    input_cost = (prompt_tokens / 1000) * COST_PER_1K_TOKENS["input"]
    output_cost = (completion_tokens / 1000) * COST_PER_1K_TOKENS["output"]
    total_cost = input_cost + output_cost

    self.log.prod({
        "cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost
    })
    return total_cost
```

#### Latency Tracking
```python
import time

async def _single_async(self, messages, **params):
    start_time = time.time()
    response = await self._make_request(messages, params)
    latency_ms = (time.time() - start_time) * 1000

    self.log.prod({
        "latency_ms": latency_ms,
        "tokens": response['usage']['total_tokens'],
        "tokens_per_second": response['usage']['total_tokens'] / (latency_ms / 1000)
    })
```

### 7. Documentation
- [ ] Update dspy README with full examples
- [ ] Add docstrings to all classes
- [ ] Create examples/ folder with:
  - `examples/simple_qa.py` - Basic Q&A
  - `examples/streaming_agent.py` - Streaming agent
  - `examples/tool_calling.py` - Agent with tools
  - `examples/multi_step.py` - Multi-step reasoning

### 8. Testing
- [ ] Add unit tests for LM class
- [ ] Add unit tests for Signature parsing
- [ ] Add integration tests with mock vLLM server
- [ ] Add benchmarks for performance tracking

## File Structure (After Tomorrow)
```
dspy/
├── __init__.py
├── lm/
│   ├── lm.py           # ✅ Updated with UniversalLogger
│   └── __init__.py
├── signature/
│   ├── signature.py
│   └── __init__.py
├── predict/
│   ├── predict.py      # ✅ Updated with UniversalLogger
│   └── __init__.py
├── agent.py            # ✅ Updated with UniversalLogger
├── tools/              # 🆕 New module
│   ├── __init__.py
│   ├── registry.py
│   ├── executor.py
│   └── builtin.py
├── evaluate/
│   └── ...
├── optimize/
│   └── ...
├── examples/           # 🆕 New folder
│   ├── simple_qa.py
│   ├── streaming_agent.py
│   ├── tool_calling.py
│   └── multi_step.py
├── tests/              # 🆕 New folder
│   ├── test_lm.py
│   ├── test_signature.py
│   └── test_agent.py
├── README.md           # ✅ Updated
└── task.md             # This file
```

## Priority Order
1. **Start vLLM server** (required for testing)
2. **Test basic functionality** (test_dspy.py)
3. **Integrate UniversalLogger** (LM, Agent, Predict)
4. **Add error handling & retry** (production-ready)
5. **Tool system** (core feature)
6. **Cost & latency tracking** (observability)
7. **Documentation & examples** (usability)
8. **Testing & benchmarks** (quality)

## Notes
- Keep it simple - no over-abstraction
- Full observability with UniversalLogger
- Built for vLLM continuous batching
- Follow FRAMEWORK_ANALYSIS.md principles
- No vendor lock-in
- Production-ready from day one

## Reference Documents
- `/src/FRAMEWORK_ANALYSIS.md` - Framework comparison & best practices
- `/src/logger/README.md` - UniversalLogger documentation
- `/src/beta/aspy/` - Original implementation (for reference)
