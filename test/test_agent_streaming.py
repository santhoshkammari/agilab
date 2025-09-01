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

def test_agent_streaming():
    """Test Agent streaming functionality."""
    print("=== Testing Agent Streaming Functionality ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[simple_calculator], stream=True, enable_rich_debug=False)
    
    try:
        print("Testing streaming agent...")
        events = []
        
        for event in agent("What's 10 + 5?"):
            print(f"Event: {event.get('type', 'unknown')} - {str(event)[:100]}...")
            events.append(event)
            
            if event.get('type') == 'final':
                break
                
            if len(events) > 20:  # Limit for testing
                break
        
        print(f"✅ Agent streaming works, received {len(events)} events")
        return True
        
    except Exception as e:
        print(f"❌ Agent streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_streaming()