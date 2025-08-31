#!/usr/bin/env python3

import sys
import os

# Add the claude directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude', 'llm'))
from __oai import OAI
from sum import sum_numbers

def test_oai_tool_call_compatibility():
    """Test that OAI tool calls work with attribute access."""
    
    print("🔍 Testing OAI tool call compatibility...")
    print("=" * 60)
    
    # Initialize OAI client
    oai = OAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    def add(a: int, b: int) -> int:
        """Add two numbers together.
        
        Args:
            a: First number
            b: Second number
        """
        return sum_numbers(a, b)
    
    try:
        print("🚀 Testing tool calling...")
        
        response = oai.chat(
            prompt="What is 25 + 15? Use the available tools.",
            tools=[add],
            model="google/gemini-2.0-flash-exp:free",
            temperature=0
        )
        
        print(f"✅ Response type: {type(response)}")
        print(f"✅ Has message: {hasattr(response, 'message')}")
        print(f"✅ Has tool_calls: {hasattr(response.message, 'tool_calls')}")
        print(f"✅ Tool calls: {response.message.tool_calls}")
        
        if response.message.tool_calls:
            for i, tool_call in enumerate(response.message.tool_calls):
                print(f"\\n🛠️  Tool Call {i+1}:")
                print(f"  Type: {type(tool_call)}")
                print(f"  Has function: {hasattr(tool_call, 'function')}")
                print(f"  Function type: {type(tool_call.function)}")
                print(f"  Function name: {tool_call.function.name}")
                print(f"  Function args: {tool_call.function.arguments}")
                print(f"  Args type: {type(tool_call.function.arguments)}")
                
                # Test the exact access pattern from input.py
                try:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    print(f"  ✅ Attribute access works: {tool_name}({tool_args})")
                    
                    # Test calling the function
                    if tool_name == "add":
                        result = add(**tool_args)
                        print(f"  ✅ Function execution: {result}")
                        
                except Exception as e:
                    print(f"  ❌ Attribute access failed: {e}")
        else:
            print("ℹ️  No tool calls in response")
        
        print("\\n🎯 Test Result: Tool call compatibility check complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    test_oai_tool_call_compatibility()