# vllm-bridge

Anthropic Messages API server for vLLM backend.

## Quick Start

```bash
./start_api.sh
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
claude --model qwen
```

## Files

- `api.py` - FastAPI server (Anthropic ↔ vLLM translation)
- `start_api.sh` - Start script
- `TOOLS_SUPPORT.md` - Tool calling documentation
- `VLLM_CLAUDE_CODE_SETUP.md` - Detailed setup guide

## Features

- Messages API compatibility
- Tool calling support
- Streaming responses
- System prompts
