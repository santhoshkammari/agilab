#!/usr/bin/env python3

import sys
import os
import json

# Add the claude directory to Python path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import OAI directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude', 'llm'))
from __oai import OAI

# Import from the main directory, not claude/llm
sys.path.insert(0, os.path.dirname(__file__))
from sum import sum_numbers
from subtract import subtract_numbers


def test_oai_with_tools():
    """Test OAI class with sum and subtract functions."""
    
    # Initialize OAI client with OpenRouter
    oai = OAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    # Define test functions
    def add(a: int, b: int) -> int:
        """Add two numbers together.
        
        Args:
            a: First number
            b: Second number
        """
        return sum_numbers(a, b)
    
    def subtract(a: int, b: int) -> int:
        """Subtract second number from first number.
        
        Args:
            a: First number
            b: Second number
        """
        return subtract_numbers(a, b)
    
    print("Testing OAI class with tools...")
    print("=" * 60)
    
    # Test 1: Simple calculation with streaming
    print("\\n🧪 Test 1: Calculate 243 + 35 (streaming)")
    print("-" * 40)
    
    try:
        accumulated_content = ""
        tool_calls_data = []
        
        for chunk in oai(
            prompt="What is 243 + 35? Use the available tools.",
            tools=[add, subtract],
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0,
            stream=True
        ):
            if "error" in chunk:
                print(f"❌ Error: {chunk['error']}")
                break
                
            if "message" in chunk:
                # Handle content
                if "content" in chunk["message"] and chunk["message"]["content"]:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    accumulated_content += content
                
                # Handle tool calls
                if "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        function_name = tool_call["function"]["name"]
                        arguments_str = tool_call["function"]["arguments"]
                        
                        # Store tool call data for execution
                        tool_calls_data.append({
                            "name": function_name,
                            "arguments": arguments_str
                        })
                        
                        print(f"\\n📞 Tool Call: {function_name}({arguments_str})")
            
            if chunk.get("done", False):
                print(f"\\n✅ Finished: {chunk.get('done_reason', 'completed')}")
                break
        
        # Execute collected tool calls
        print("\\n🔧 Executing tool calls:")
        for tool_call in tool_calls_data:
            try:
                if tool_call["arguments"]:
                    args = json.loads(tool_call["arguments"])
                    if tool_call["name"] == "add":
                        result = add(**args)
                        print(f"  add({args}) = {result}")
                    elif tool_call["name"] == "subtract":
                        result = subtract(**args)
                        print(f"  subtract({args}) = {result}")
            except Exception as e:
                print(f"  ❌ Error executing {tool_call['name']}: {e}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Non-streaming mode
    print("\\n\\n🧪 Test 2: Calculate 100 - 25 (non-streaming)")
    print("-" * 40)
    
    try:
        response = oai.chat(
            prompt="What is 100 - 25? Use the available tools.",
            tools=[add, subtract],
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0
        )
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            print(f"✅ Response: {response}")
            
            # Execute tool calls if any
            if "message" in response and "tool_calls" in response["message"]:
                print("\\n🔧 Executing tool calls:")
                for tool_call in response["message"]["tool_calls"]:
                    try:
                        function_name = tool_call["function"]["name"]
                        args = json.loads(tool_call["function"]["arguments"])
                        
                        if function_name == "add":
                            result = add(**args)
                            print(f"  add({args}) = {result}")
                        elif function_name == "subtract":
                            result = subtract(**args)
                            print(f"  subtract({args}) = {result}")
                    except Exception as e:
                        print(f"  ❌ Error executing {function_name}: {e}")
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Multiple operations
    print("\\n\\n🧪 Test 3: Multiple operations (150 + 50, then subtract 75)")
    print("-" * 40)
    
    try:
        accumulated_content = ""
        tool_calls_data = []
        
        for chunk in oai(
            prompt="Calculate 150 + 50, then subtract 75 from that result. Show your work step by step.",
            tools=[add, subtract],
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0,
            stream=True
        ):
            if "error" in chunk:
                print(f"❌ Error: {chunk['error']}")
                break
                
            if "message" in chunk:
                # Handle content
                if "content" in chunk["message"] and chunk["message"]["content"]:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    accumulated_content += content
                
                # Handle tool calls
                if "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        function_name = tool_call["function"]["name"]
                        arguments_str = tool_call["function"]["arguments"]
                        
                        tool_calls_data.append({
                            "name": function_name,
                            "arguments": arguments_str
                        })
                        
                        print(f"\\n📞 Tool Call: {function_name}({arguments_str})")
            
            if chunk.get("done", False):
                print(f"\\n✅ Finished: {chunk.get('done_reason', 'completed')}")
                break
        
        # Execute collected tool calls
        print("\\n🔧 Executing tool calls:")
        results = []
        for tool_call in tool_calls_data:
            try:
                if tool_call["arguments"]:
                    args = json.loads(tool_call["arguments"])
                    if tool_call["name"] == "add":
                        result = add(**args)
                        results.append(result)
                        print(f"  add({args}) = {result}")
                    elif tool_call["name"] == "subtract":
                        result = subtract(**args)
                        results.append(result)
                        print(f"  subtract({args}) = {result}")
            except Exception as e:
                print(f"  ❌ Error executing {tool_call['name']}: {e}")
        
        if results:
            print(f"\\n🎯 Final results: {results}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    print("\\n" + "=" * 60)
    print("🏁 OAI Testing Complete!")


if __name__ == "__main__":
    test_oai_with_tools()