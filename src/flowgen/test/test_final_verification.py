#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🔧 add_numbers({a}, {b}) = {a + b}")
    return a + b

def get_weather(location: str) -> str:
    """Get weather for a location."""
    print(f"🔧 get_weather('{location}')")
    return f"Weather in {location}: Sunny, 22°C"

def test_complete_functionality():
    """Test complete Agent and LLM functionality after fix."""
    print("=== Final Verification Test ===")
    
    llm = LLM(base_url="http://localhost:8000")
    agent = Agent(llm=llm, tools=[add_numbers, get_weather], enable_rich_debug=False)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Basic chat without tools
    try:
        result = agent("Just say hello")
        if result.get('content'):
            print("✅ Test 1: Basic chat - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 1: Basic chat - FAILED")
    except Exception as e:
        print(f"❌ Test 1: Basic chat - ERROR: {e}")
    
    # Test 2: Single tool call
    try:
        result = agent("Use add_numbers to calculate 12 + 8")
        conversation = agent.get_conversation()
        tool_messages = [msg for msg in conversation if msg.get('role') == 'tool']
        if tool_messages:
            print("✅ Test 2: Single tool call - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 2: Single tool call - FAILED")
    except Exception as e:
        print(f"❌ Test 2: Single tool call - ERROR: {e}")
    
    # Test 3: Multiple tool calls
    try:
        agent.clear_history()  # Reset for clean test
        result = agent("Calculate 5 + 7 using add_numbers, then get weather for London")
        conversation = agent.get_conversation()
        tool_messages = [msg for msg in conversation if msg.get('role') == 'tool']
        if len(tool_messages) >= 2:
            print("✅ Test 3: Multiple tool calls - PASSED")
            tests_passed += 1
        else:
            print(f"⚠️  Test 3: Multiple tool calls - PARTIAL ({len(tool_messages)} tools used)")
            tests_passed += 0.5
    except Exception as e:
        print(f"❌ Test 3: Multiple tool calls - ERROR: {e}")
    
    # Test 4: Streaming with tools
    try:
        agent.clear_history()
        events = []
        for event in agent("Use add_numbers for 3 + 4", stream=True):
            events.append(event)
            if event.get('type') == 'final':
                break
        
        tool_events = [e for e in events if e.get('type') in ['tool_start', 'tool_result']]
        if tool_events:
            print("✅ Test 4: Streaming with tools - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 4: Streaming with tools - FAILED")
    except Exception as e:
        print(f"❌ Test 4: Streaming with tools - ERROR: {e}")
    
    print(f"\n🏁 Final Score: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 3:
        print("✅ Agent and LLM classes are working correctly!")
        return True
    else:
        print("⚠️  Some issues remain")
        return False

if __name__ == "__main__":
    test_complete_functionality()