#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgen.llm.llm import LLM
import json
import time

def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 22°C, sunny, humidity 65%"

def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers"""
    return a + b

def test_basic_chat():
    """Test 1: Basic chat conversation"""
    print("\n=== Test 1: Basic Chat ===")
    
    llm = LLM()
    
    response = llm.chat([
        {"role": "user", "content": "Hello! What's the capital of France?"}
    ], max_tokens=50, temperature=0.1)
    
    print(f"Response: {json.dumps(response, indent=2)}")
    print(f"Content: {response.get('content', '')}")
    
    return response

def test_multiturn_conversation():
    """Test 2: Multi-turn conversation with message building"""
    print("\n=== Test 2: Multi-turn Conversation ===")
    
    llm = LLM()
    
    messages = [
        {"role": "user", "content": "I'm planning a trip to Japan."}
    ]
    
    # First exchange
    response1 = llm.chat(messages, max_tokens=100, temperature=0.3)
    print(f"Assistant 1: {response1.get('content', '')}")
    
    # Build conversation
    messages.append({"role": "assistant", "content": response1.get('content', '')})
    messages.append({"role": "user", "content": "What's the best time to visit for cherry blossoms?"})
    
    # Second exchange
    response2 = llm.chat(messages, max_tokens=100, temperature=0.3)
    print(f"Assistant 2: {response2.get('content', '')}")
    
    # Continue building
    messages.append({"role": "assistant", "content": response2.get('content', '')})
    messages.append({"role": "user", "content": "Thank you! Any food recommendations?"})
    
    # Third exchange
    response3 = llm.chat(messages, max_tokens=100, temperature=0.3)
    print(f"Assistant 3: {response3.get('content', '')}")
    
    print(f"Final conversation length: {len(messages)} messages")
    return messages

def test_structured_json():
    """Test 3: Structured JSON response format"""
    print("\n=== Test 3: Structured JSON Response ===")
    
    llm = LLM()
    
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "number"},
            "famous_for": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["city", "country"]
    }
    
    response = llm.chat([
        {"role": "user", "content": "Tell me about Tokyo in the specified JSON format"}
    ], format=schema, max_tokens=200, temperature=0.1)
    
    print(f"JSON Response: {json.dumps(response, indent=2)}")
    
    content = response.get('content', '')
    if content:
        try:
            parsed_json = json.loads(content)
            print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw content: {content}")
    
    return response

def test_tool_calling():
    """Test 4: Tool calling with get_weather and calculate_sum"""
    print("\n=== Test 4: Tool Calling ===")
    
    llm = LLM()
    
    response = llm.chat([
        {"role": "user", "content": "What's the weather in Tokyo and what's 15 + 27?"}
    ], tools=[get_weather, calculate_sum], max_tokens=300, temperature=0.1)
    
    print(f"Tool Call Response: {json.dumps(response, indent=2)}")
    
    # Check for tool calls
    tool_calls = response.get('tool_calls', [])
    if tool_calls:
        print(f"\nTool calls found: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            func_info = tool_call.get('function', {})
            print(f"Tool call {i+1}: {func_info.get('name', 'unknown')} with args {func_info.get('arguments', {})}")
    else:
        print("No tool calls detected in response")
        content = response.get('content', '')
        print(f"Raw response: {content}")
    
    return response

def main():
    print("Testing Flowgen LLM with different scenarios")
    
    # Run all tests
    try:
        test_basic_chat()
        time.sleep(1)
        
        test_multiturn_conversation()
        time.sleep(1)
        
        test_structured_json()
        time.sleep(1)
        
        test_tool_calling()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== LLM Testing Complete ===")

if __name__ == "__main__":
    main()