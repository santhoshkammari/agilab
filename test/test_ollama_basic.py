#!/usr/bin/env python3
"""
Test basic Ollama LLM functionality
"""

import json
from claude.llm import OAI

def test_basic_chat():
    """Test basic chat without tools"""
    print("=== Testing Basic Chat ===")
    try:
        llm = OAI(provider="ollama", base_url="http://localhost:11434/v1", model="qwen3:0.6b")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = llm.chat(messages)
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error in basic chat: {e}")
        return False

def test_streaming_chat():
    """Test streaming chat"""
    print("\n=== Testing Streaming Chat ===")
    try:
        llm = OAI(provider="ollama", base_url="http://localhost:11434/v1", model="qwen3:0.6b")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5"}
        ]
        
        stream = llm.stream_chat(messages)
        print("Streaming response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"Error in streaming chat: {e}")
        return False

def test_tool_calling():
    """Test tool calling with sum/subtract functions"""
    print("\n=== Testing Tool Calling ===")
    try:
        llm = OAI(provider="ollama", base_url="http://localhost:11434/v1", model="qwen3:0.6b")
        
        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "sum_numbers",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "subtract_numbers",
                    "description": "Subtract second number from first number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "What is 15 + 7?"}
        ]
        
        response = llm.tool_call(messages, tools)
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                
                # Execute the tool call
                if tool_call.function.name == "sum_numbers":
                    args = json.loads(tool_call.function.arguments)
                    result = args["a"] + args["b"]
                    print(f"  Result: {result}")
                elif tool_call.function.name == "subtract_numbers":
                    args = json.loads(tool_call.function.arguments)
                    result = args["a"] - args["b"]
                    print(f"  Result: {result}")
        else:
            print("No tool calls detected")
            print(f"Response: {response.choices[0].message.content}")
        
        return True
    except Exception as e:
        print(f"Error in tool calling: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Ollama LLM Integration")
    print("=" * 40)
    
    results = {
        "basic_chat": test_basic_chat(),
        "streaming_chat": test_streaming_chat(), 
        "tool_calling": test_tool_calling()
    }
    
    print(f"\n=== Test Results ===")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    main()