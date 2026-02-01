#!/bin/bash

# Test the Anthropic Messages API server

PORT=${1:-1234}
VLLM_URL=${2:-"http://192.168.170.76:8000"}

echo "Testing Anthropic Messages API Server"
echo "======================================"
echo "Port: $PORT"
echo "vLLM URL: $VLLM_URL"
echo ""

# Kill any existing servers on this port
fuser -k $PORT/tcp 2>/dev/null || true
sleep 1

# Start the server
echo "Starting API server..."
python api.py --port $PORT --vllm-url "$VLLM_URL" > /tmp/api_test.log 2>&1 &
API_PID=$!
sleep 4

echo "API PID: $API_PID"
echo ""

# Test health endpoint
echo "Testing health endpoint..."
HEALTH=$(curl -s http://localhost:$PORT/health)
echo "$HEALTH" | python -m json.tool 2>/dev/null || echo "$HEALTH"
echo ""

# Test models endpoint
echo "Testing models endpoint..."
MODELS=$(curl -s http://localhost:$PORT/v1/models)
echo "$MODELS" | python -m json.tool 2>/dev/null | head -20
echo ""

# Test messages endpoint
echo "Testing messages endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:$PORT/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer lmstudio" \
  -d '{
    "model": "qwen",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Say hi"}
    ]
  }')

echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Clean up
echo "Cleaning up..."
kill $API_PID 2>/dev/null || true
sleep 1

echo "Done!"
