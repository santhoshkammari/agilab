#!/usr/bin/env python3
"""
Test script for tools support in api.py
"""

import requests
import json

BASE_URL = "http://localhost:1234"

def test_basic_tool_call():
    """Test basic tool call (non-streaming)"""
    print("\n=== Test 1: Basic Tool Call (Non-Streaming) ===")

    payload = {
        "model": "qwen",
        "max_tokens": 1024,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }],
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
    }

    response = requests.post(f"{BASE_URL}/v1/messages", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Stop reason: {result.get('stop_reason')}")
        print(f"✓ Content blocks: {len(result.get('content', []))}")
        for i, block in enumerate(result.get('content', [])):
            print(f"  Block {i}: type={block.get('type')}, ", end="")
            if block.get('type') == 'tool_use':
                print(f"name={block.get('name')}, id={block.get('id')}")
            else:
                print(f"text={block.get('text', '')[:50]}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_tool_result():
    """Test tool result handling"""
    print("\n=== Test 2: Tool Result Handling ===")

    payload = {
        "model": "qwen",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_test123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_test123",
                    "content": "72°F and sunny"
                }]
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/v1/messages", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Stop reason: {result.get('stop_reason')}")
        print(f"✓ Response: {result.get('content', [{}])[0].get('text', '')[:100]}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_server_info():
    """Test server info endpoint"""
    print("\n=== Test 3: Server Info ===")

    response = requests.get(f"{BASE_URL}/")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Name: {result.get('name')}")
        print(f"✓ Version: {result.get('version')}")
        print(f"✓ Features: {', '.join(result.get('features', []))}")
    else:
        print(f"✗ Error: {response.status_code}")


def test_health():
    """Test health endpoint"""
    print("\n=== Test 4: Health Check ===")

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Status: {result.get('status')}")
        print(f"✓ vLLM: {result.get('vllm')}")
    else:
        print(f"✗ Error: {response.status_code}")


if __name__ == "__main__":
    print("Testing Tools Support in api.py")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")

    try:
        test_server_info()
        test_health()
        test_basic_tool_call()
        test_tool_result()

        print("\n" + "=" * 50)
        print("Tests completed!")

    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to server. Make sure api.py is running:")
        print("  python api.py --port 1234")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
