#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    print(f"🔧 calculate({expression})")
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

def get_weather(location: str) -> str:
    """Get weather for a location."""
    print(f"🔧 get_weather({location})")
    return f"Weather in {location}: Sunny, 22°C"

def test_agent_explicit_tools():
    """Test Agent with explicit tool calling instructions."""
    print("=== Testing Agent with Explicit Tool Instructions ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[calculate, get_weather], enable_rich_debug=False)
    
    try:
        # Test with explicit tool instruction
        prompt = "You have access to calculate and get_weather functions. Use the calculate function to find 12 + 8, then use get_weather to check Paris weather."
        
        result = agent(prompt)
        
        print(f"Result: {result}")
        
        # Check conversation for tool usage
        conversation = agent.get_conversation()
        tool_messages = [msg for msg in conversation if msg.get('role') == 'tool']
        
        print(f"\nTool messages found: {len(tool_messages)}")
        for i, tool_msg in enumerate(tool_messages):
            print(f"  {i+1}. {tool_msg.get('name')}: {tool_msg.get('content')}")
            
        if len(tool_messages) >= 2:
            print("✅ Multiple tool calls working correctly")
        elif len(tool_messages) == 1:
            print("⚠️  Only one tool call detected")
        else:
            print("❌ No tool calls detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Agent explicit tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_explicit_tools()