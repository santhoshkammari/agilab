# FlowGen API

A FastAPI-based service for running local LLaMA models with tool calling and JSON schema support.

## Features

- **Model Loading**: Load and unload LLaMA models dynamically
- **Chat Completions**: Generate responses with tool calling support
- **JSON Schema**: Enforce structured output with Pydantic models
- **Streaming**: Real-time response streaming
- **Tool Parsing**: Multiple tool call formats supported

## Quick Start

1. Start the server:
```bash
python main.py
```
Server runs on `http://localhost:8001`

2. Load a model:
```bash
curl -X POST http://localhost:8001/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/your/model.gguf",
    "tokenizer_name": "LiquidAI/LFM2-350M"
  }'
```

3. Start chatting:
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## API Endpoints

### POST /load

Load a model and tokenizer.

**Parameters:**
- `model_path` (string, required): Path to the GGUF model file
- `tokenizer_name` (string, required): HuggingFace tokenizer name
- `options` (object, optional): LLaMA model options

**Default options:**
```json
{
  "n_ctx": 2048,
  "n_batch": 256,
  "verbose": false,
  "n_gpu_layers": 0
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8001/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/home/user/models/model.gguf",
    "tokenizer_name": "LiquidAI/LFM2-350M",
    "options": {
      "n_ctx": 4096,
      "n_batch": 512
    }
  }'
```

**Python Example:**
```python
import requests

response = requests.post("http://localhost:8001/load", json={
    "model_path": "/path/to/model.gguf",
    "tokenizer_name": "LiquidAI/LFM2-350M",
    "options": {
        "n_ctx": 4096,
        "temperature": 0.7
    }
})
print(response.json())
```

### POST /chat

Generate chat completions with optional tool calling and JSON schema.

**Parameters:**
- `messages` (array, required): Chat messages in OpenAI format
- `tools` (array, optional): Tool definitions for function calling
- `options` (object, optional): Generation options
- `response_format` (object, optional): JSON schema for structured output

**Default options:**
```json
{
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 40,
  "stream": false,
  "repeat_penalty": 1.1
}
```

**Basic Chat cURL:**
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "options": {
      "max_tokens": 50,
      "temperature": 0.1
    }
  }'
```

**Tool Calling cURL:**
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Calculate 15 + 25"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate_sum",
          "description": "Calculates the sum of two integers",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {"type": "integer"},
              "b": {"type": "integer"}
            },
            "required": ["a", "b"]
          }
        }
      }
    ]
  }'
```

**JSON Schema cURL:**
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about a basketball team"}
    ],
    "response_format": {
      "type": "json_object",
      "schema": {
        "type": "object",
        "properties": {
          "team_name": {"type": "string"},
          "score": {"type": "integer"},
          "city": {"type": "string"}
        },
        "required": ["team_name", "score", "city"]
      }
    }
  }'
```

**Streaming cURL:**
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "options": {
      "stream": true,
      "max_tokens": 50
    }
  }'
```

### POST /unload

Unload the current model and tokenizer.

**cURL Example:**
```bash
curl -X POST http://localhost:8001/unload
```

## Python Examples

### Basic Chat
```python
import requests

# Load model first
requests.post("http://localhost:8001/load", json={
    "model_path": "/path/to/model.gguf",
    "tokenizer_name": "LiquidAI/LFM2-350M"
})

# Chat
response = requests.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
})

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Tool Calling
```python
import requests
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = requests.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    "tools": tools
})

result = response.json()
print(f"Response: {result['choices'][0]['message']['content']}")
print(f"Tool calls: {result['tool_calls']}")
```

### JSON Schema Enforcement
```python
import requests
from pydantic import BaseModel

class TeamInfo(BaseModel):
    team_name: str
    score: int
    city: str

response = requests.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "Tell me about a basketball team"}
    ],
    "response_format": {
        "type": "json_object",
        "schema": TeamInfo.model_json_schema()
    }
})

result = response.json()
content = result["choices"][0]["message"]["content"]
team_data = TeamInfo(**json.loads(content))
print(team_data)
```

### Streaming Response
```python
import requests
import json

response = requests.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "Count from 1 to 10"}
    ],
    "options": {
        "stream": True,
        "max_tokens": 100
    }
}, stream=True)

print("Streaming response: ", end="")
for line in response.iter_lines():
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data:"):
            data = line_str[5:].strip()
            if data == "[DONE]":
                break
            
            try:
                chunk_data = json.loads(data)
                if 'content' in chunk_data:
                    print(chunk_data['content'], end="", flush=True)
                elif 'completion' in chunk_data:
                    completion = chunk_data['completion']
                    tool_calls = completion.get('tool_calls', [])
                    if tool_calls:
                        print(f"\nTool calls: {tool_calls}")
            except:
                pass
print()
```

## Tool Call Formats

The API supports multiple tool call formats in model responses:

### Format 1: Function List
```
[calculate_sum(a=15, b=25), get_weather(location="Paris")]
```

### Format 2: XML-style
```xml
<tool_call>{"name": "calculate_sum", "arguments": {"a": 15, "b": 25}}</tool_call>
<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>
```

## Response Format

### Non-streaming Response
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The sum is 40."
      },
      "finish_reason": "stop"
    }
  ],
  "tool_calls": [
    {
      "name": "calculate_sum",
      "arguments": {"a": 15, "b": 25}
    }
  ]
}
```

### Streaming Response
```
data: {"content": "The"}
data: {"content": " sum"}
data: {"content": " is 40."}
data: {"completion": {"choices": [...], "tool_calls": [...]}}
data: [DONE]
```

## Available Options

### Model Loading Options
- `model_path`: Path to the GGUF model file (required)
- `n_gpu_layers`: Number of layers to offload to GPU (default: 0, -1 for all layers)
- `split_mode`: How to split the model across GPUs (default: 1)
- `main_gpu`: The GPU that is used for the entire model (default: 0)
- `tensor_split`: How split tensors should be distributed across GPUs
- `vocab_only`: Only load the vocabulary no weights (default: false)
- `use_mmap`: Use mmap if possible (default: true)
- `use_mlock`: Force the system to keep the model in RAM (default: false)
- `kv_overrides`: Key-value overrides for the model
- `seed`: RNG seed, -1 for random (default: 4294967295)
- `n_ctx`: Text context, 0 = from model (default: 2048)
- `n_batch`: Prompt processing maximum batch size (default: 256)
- `n_ubatch`: Physical batch size (default: 512)
- `n_threads`: Number of threads to use for generation
- `n_threads_batch`: Number of threads to use for batch processing
- `rope_scaling_type`: RoPE scaling type (default: -1)
- `pooling_type`: Pooling type (default: -1)
- `rope_freq_base`: RoPE base frequency, 0 = from model (default: 0.0)
- `rope_freq_scale`: RoPE frequency scaling factor, 0 = from model (default: 0.0)
- `yarn_ext_factor`: YaRN extrapolation mix factor, negative = from model (default: -1.0)
- `yarn_attn_factor`: YaRN magnitude scaling factor (default: 1.0)
- `yarn_beta_fast`: YaRN low correction dim (default: 32.0)
- `yarn_beta_slow`: YaRN high correction dim (default: 1.0)
- `yarn_orig_ctx`: YaRN original context size (default: 0)
- `logits_all`: Return logits for all tokens, not just the last token (default: false)
- `embedding`: Embedding mode only (default: false)
- `offload_kqv`: Offload K, Q, V to GPU (default: true)
- `flash_attn`: Use flash attention (default: false)
- `op_offload`: Offload host tensor operations to device
- `swa_full`: Use full-size SWA cache
- `no_perf`: Measure performance timings (default: false)
- `last_n_tokens_size`: Maximum number of tokens to keep in the last_n_tokens deque (default: 64)
- `lora_base`: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model
- `lora_scale`: LoRA scaling factor (default: 1.0)
- `lora_path`: Path to a LoRA file to apply to the model
- `numa`: NUMA policy (default: false)
- `chat_format`: String specifying the chat format to use when calling create_chat_completion
- `chat_handler`: Optional chat handler to use when calling create_chat_completion
- `draft_model`: Optional draft model to use for speculative decoding
- `tokenizer`: Optional tokenizer to override the default tokenizer from llama.cpp
- `verbose`: Print verbose output to stderr (default: false)
- `type_k`: KV cache data type for K (default: f16)
- `type_v`: KV cache data type for V (default: f16)
- `spm_infill`: Use Suffix/Prefix/Middle pattern for infill (default: false)

### Chat Completion Options
- `prompt`: The input prompt (automatically generated from messages)
- `suffix`: A suffix to append to the generated text
- `max_tokens`: Maximum tokens to generate (default: 100, unlimited if <= 0)
- `temperature`: Sampling temperature (default: 0.8)
- `top_p`: Top-p value for nucleus sampling (default: 0.95)
- `min_p`: Min-p value for minimum p sampling (default: 0.05)
- `typical_p`: Typical-p value for locally typical sampling (default: 1.0)
- `logprobs`: Number of logprobs to return
- `echo`: Whether to echo the prompt (default: false)
- `stop`: Stop sequences (string or array of strings, default: [])
- `frequency_penalty`: Penalty for token frequency (default: 0.0)
- `presence_penalty`: Penalty for token presence (default: 0.0)
- `repeat_penalty`: Repetition penalty (default: 1.0)
- `top_k`: Top-k value for sampling (default: 40)
- `stream`: Enable streaming (default: false)
- `seed`: Seed for sampling
- `tfs_z`: Tail-free sampling parameter (default: 1.0)
- `mirostat_mode`: Mirostat sampling mode (default: 0)
- `mirostat_tau`: Target cross-entropy for mirostat (default: 5.0)
- `mirostat_eta`: Learning rate for mirostat (default: 0.1)
- `model`: Model name for the completion object
- `stopping_criteria`: Custom stopping criteria
- `logits_processor`: Custom logits processors
- `grammar`: Grammar constraints for structured output
- `logit_bias`: Logit bias dictionary

## Testing

Run the included test suite:
```bash
python infer.py
```

This tests all endpoints with various configurations including tool calling, JSON schema, and streaming.

## Error Handling

All endpoints return error information in this format:
```json
{
  "error": "Error description"
}
```

Common errors:
- "Model not loaded. Call /load first." - Load a model before chatting
- Model file not found - Check the model_path
- Invalid tokenizer - Verify the tokenizer_name exists on HuggingFace