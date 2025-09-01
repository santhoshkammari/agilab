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

def get_weather(location: str) -> str:
    """Get weather for a location."""
    print(f"🔧 Tool called: get_weather('{location}')")
    return f"Weather in {location}: Sunny, 22°C"

def test_agent_with_sufficient_tokens():
    """Test Agent with sufficient max_tokens for tool calls."""
    print("=== Testing Agent with Sufficient Tokens ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[add_numbers, get_weather], enable_rich_debug=False)
    
    try:
        # Test with higher max_tokens
        result = agent("Use add_numbers to calculate 15 + 25", max_tokens=1000)
        
        print(f"Result: {result}")
        
        # Check conversation for tool usage
        conversation = agent.get_conversation()
        tool_messages = [msg for msg in conversation if msg.get('role') == 'tool']
        
        print(f"\nTool messages found: {len(tool_messages)}")
        for tool_msg in tool_messages:
            print(f"  - {tool_msg.get('name')}: {tool_msg.get('content')}")
            
        if tool_messages:
            print("✅ Tool calls working with sufficient tokens")
        else:
            print("❌ Still no tool calls")
            
        return len(tool_messages) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_with_sufficient_tokens()