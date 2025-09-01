#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🔧 Tool called: add_numbers({a}, {b})")
    return a + b

def get_info(topic: str) -> str:
    """Get information about a topic."""
    print(f"🔧 Tool called: get_info('{topic}')")
    return f"Information about {topic}: This is a test response."

def test_agent_with_tools():
    """Test Agent with actual tool execution."""
    print("=== Testing Agent with Tool Calls ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[add_numbers, get_info], enable_rich_debug=False, max_iterations=5)
    
    try:
        print("Testing agent with explicit tool request...")
        result = agent("Use the add_numbers tool to calculate 7 + 9")
        
        print(f"Final result: {result}")
        print(f"Iterations used: {result.get('iterations', 0)}")
        
        # Check if tools were actually called
        conversation = agent.get_conversation()
        tool_messages = [msg for msg in conversation if msg.get('role') == 'tool']
        print(f"Tool messages found: {len(tool_messages)}")
        
        if tool_messages:
            print("✅ Tool calls working correctly")
            for tool_msg in tool_messages:
                print(f"  Tool {tool_msg.get('name')}: {tool_msg.get('content')}")
        else:
            print("⚠️  No tool calls detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Agent with tools failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_with_tools()