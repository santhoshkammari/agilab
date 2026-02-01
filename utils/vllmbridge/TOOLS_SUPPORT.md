# Tools Support Implementation

## Overview
The `api.py` server now supports full Anthropic-compatible tool calling, bridging between Claude Code's tool format and vLLM's OpenAI-compatible tool format.

## What Was Added

### 1. Data Models
- Extended `ContentBlock` to support tool_use blocks with `id`, `name`, and `input` fields
- Added `tools` and `tool_choice` fields to `MessagesRequest`

### 2. Conversion Functions
- `convert_anthropic_tools_to_openai()` - Converts Anthropic tool definitions to OpenAI format
- `convert_tool_choice_to_openai()` - Converts tool_choice parameter (auto/any/tool/none)
- `convert_openai_tool_calls_to_anthropic()` - Converts vLLM tool_calls back to Anthropic tool_use blocks

### 3. Message Processing
- Handles `tool_use` blocks in assistant messages (converts to OpenAI tool_calls)
- Handles `tool_result` blocks in user messages (converts to OpenAI tool messages)
- Properly reconstructs multi-turn conversations with tools

### 4. Response Handling
- **Non-streaming**: Detects tool_calls in vLLM response and converts to tool_use content blocks
- **Streaming**: Accumulates tool call deltas and emits proper Anthropic streaming events
  - `content_block_start` for tool_use blocks
  - `content_block_delta` with `input_json_delta` for tool arguments
  - `content_block_stop` when tool call completes

### 5. Stop Reason Mapping
- Maps vLLM's `tool_calls` finish reason to Anthropic's `tool_use` stop reason

## Format Translation

### Anthropic → OpenAI (Request)
```python
# Anthropic format (from Claude Code)
{
  "tools": [{
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {...}
  }],
  "tool_choice": {"type": "auto"}
}

# ↓ Converted to ↓

# OpenAI format (to vLLM)
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {...}
    }
  }],
  "tool_choice": "auto"
}
```

### OpenAI → Anthropic (Response)
```python
# OpenAI format (from vLLM)
{
  "message": {
    "tool_calls": [{
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": '{"location": "SF"}'
      }
    }]
  },
  "finish_reason": "tool_calls"
}

# ↓ Converted to ↓

# Anthropic format (to Claude Code)
{
  "content": [{
    "type": "tool_use",
    "id": "call_123",
    "name": "get_weather",
    "input": {"location": "SF"}
  }],
  "stop_reason": "tool_use"
}
```

## Tool Choice Mappings
- `"auto"` → `"auto"` (let model decide)
- `"any"` → `"required"` (force tool use)
- `"none"` → `"none"` (no tools)
- `{"type": "tool", "name": "X"}` → `{"type": "function", "function": {"name": "X"}}` (specific tool)

## Testing

### Run the API Server
```bash
python api.py --port 1234
```

### Run Tests
```bash
python test_tools.py
```

### Manual Test with curl
```bash
curl http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "max_tokens": 1024,
    "tools": [{
      "name": "calculator",
      "description": "Perform calculations",
      "input_schema": {
        "type": "object",
        "properties": {
          "expression": {"type": "string"}
        },
        "required": ["expression"]
      }
    }],
    "messages": [{"role": "user", "content": "What is 25 * 4?"}]
  }'
```

### Test with Claude Code
```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
claude --model qwen
```

## Compatibility Notes

### vLLM Requirements
- Your vLLM instance must be running a model that supports tool calling
- The model should support the OpenAI tool calling format
- Examples: Qwen2.5, Llama 3.1+, Mistral models with tool support

### Known Limitations
- Depends on vLLM's tool calling implementation
- Some models may not support parallel tool calls
- Tool validation depends on vLLM's schema validation

## Files Modified
- `api.py` - Main implementation (v2.0.0)

## Files Created
- `test_tools.py` - Test script for tools support
- `TOOLS_SUPPORT.md` - This documentation
