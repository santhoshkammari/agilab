#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent

def simple_calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 22°C"

def test_agent_basic():
    """Test basic Agent functionality."""
    print("=== Testing Agent Basic Functionality ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[simple_calculator, get_weather], enable_rich_debug=False)
    
    try:
        # Test agent without tool calls
        print("1. Testing agent without tools...")
        result = agent("Just say hello")
        print(f"✅ Basic agent call successful")
        print(f"Response: {result.get('content', '')[:100]}...")
        
        # Test agent with tool calls
        print("\n2. Testing agent with tool calls...")
        result = agent("What's 5 + 3 and what's the weather in Paris?")
        print(f"✅ Tool-calling agent successful")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Response: {result.get('content', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_basic()