# vLLM + Claude Code Setup Guide

This guide shows how to use Claude Code with your local vLLM server (Qwen3-14B at `192.168.170.76:8000`).

## What This Does

The `api.py` server implements the **Anthropic Messages API** and routes requests to your vLLM backend. This is similar to how LM Studio provides an Anthropic-compatible `/v1/messages` endpoint.

## Quick Start

### 1. Start the API Server

```bash
./start_api.sh
```

The server will start on `http://localhost:1234` and connect to your vLLM instance at `192.168.170.76:8000`.

### 2. Configure Claude Code (Terminal)

In a new terminal, set environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
```

### 3. Use Claude Code

```bash
claude --model qwen
```

Or use with specific files:

```bash
claude read src/main.py
claude write src/config.py
```

## Configuration for VS Code

If you're using the Claude Code extension in VS Code, add to your `.vscode/settings.json`:

```json
{
  "claudeCode.environmentVariables": [
    {
      "name": "ANTHROPIC_BASE_URL",
      "value": "http://localhost:1234"
    },
    {
      "name": "ANTHROPIC_AUTH_TOKEN",
      "value": "lmstudio"
    }
  ]
}
```

## Advanced Usage

### Custom Port

```bash
python api.py --port 5000
```

Then set: `export ANTHROPIC_BASE_URL=http://localhost:5000`

### Remote vLLM Server

```bash
python api.py --vllm-url http://192.168.170.76:8000 --port 1234
```

### Check Server Health

```bash
curl http://localhost:1234/health
```

### List Available Models

```bash
curl http://localhost:1234/v1/models
```

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/messages` | Create messages (Claude API compatible) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /` | Server info |

## Example: Direct API Call

```bash
curl http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer lmstudio" \
  -d '{
    "model": "qwen",
    "max_tokens": 512,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Environment Variables

- `VLLM_BASE_URL`: vLLM server URL (default: `http://192.168.170.76:8000`)
- `VLLM_TIMEOUT`: Request timeout in seconds (default: `300`)

Set before starting the server:

```bash
export VLLM_BASE_URL=http://192.168.170.76:8000
export VLLM_TIMEOUT=300
python api.py --port 1234
```

## Troubleshooting

### Server won't start
- Check port 1234 is available: `netstat -tuln | grep 1234`
- Use different port: `python api.py --port 5000`

### Can't connect to vLLM
- Verify vLLM is running: `curl http://192.168.170.76:8000/health`
- Check network: `ping 192.168.170.76`

### Model not found
- List available models: `curl http://localhost:1234/v1/models`
- Use the full model path from the response

### Slow responses
- Check vLLM load: `curl http://192.168.170.76:8000/v1/models`
- Increase timeout: `export VLLM_TIMEOUT=600`

## Technical Details

### Supported Features

✅ Messages API (full compatibility)
✅ Multi-turn conversations
✅ System prompts
✅ Temperature & sampling parameters
✅ Token counting
✅ Model normalization
✅ Streaming responses (SSE)

### Message Flow

```
Claude Code
    ↓
curl/Anthropic SDK
    ↓
http://localhost:1234/v1/messages
    ↓
api.py (this server)
    ↓
http://192.168.170.76:8000/v1/chat/completions
    ↓
vLLM + Qwen3-14B
    ↓
Response (converted to Claude format)
```

## Performance Notes

- Context size: 50,000 tokens (Qwen3-14B)
- Recommended min: 25K tokens context
- Model: Qwen3-14B (14 billion parameters)
- vLLM optimizations: paged attention, request batching

## Next Steps

1. Start the API server: `./start_api.sh`
2. In another terminal, set env vars and run Claude Code
3. Try it: `claude --model qwen "explain machine learning"`


==============================
commands
=================================

export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
claude --model qwen --disallowedTools "Task,TaskOutput,ExitPlanMode,NotebookEdit,WebSearch,TaskStop,AskUserQuestion,Skill,EnterPlanMode,TaskCreate,TaskGet,TaskUpdate,TaskList,mcp__MultiEdit__replace_str_in_file,ListMcpResourcesTool,ReadMcpResourceTool,WebFetch" --dangerously-skip-permissions 
