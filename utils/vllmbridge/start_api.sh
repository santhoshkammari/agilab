#!/bin/bash

# Start vLLM Messages API server for Claude Code
# This mimics LM Studio's /v1/messages endpoint

echo "Starting Anthropic Messages API server for vLLM..."
echo ""
echo "To use with Claude Code, set these environment variables:"
echo ""
echo "  export ANTHROPIC_BASE_URL=http://localhost:1234"
echo "  export ANTHROPIC_AUTH_TOKEN=lmstudio"
echo ""
echo "Then run Claude Code:"
echo "  claude --model qwen"
echo ""
echo "Or set in VS Code settings for Claude Code extension:"
echo '  "claudeCode.environmentVariables": ['
echo '    {"name": "ANTHROPIC_BASE_URL", "value": "http://localhost:1234"},'
echo '    {"name": "ANTHROPIC_AUTH_TOKEN", "value": "lmstudio"}'
echo '  ]'
echo ""
echo "Starting server on port 1234..."
echo ""

python api.py --port 1234 --vllm-url http://192.168.170.76:8000
