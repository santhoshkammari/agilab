#!/usr/bin/env python3
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import vLLM
from flowgen.agent.agent import Agent

# Example tools for testing
def get_weather(location: str) -> str:
    """Get current weather for a location"""
    return f"Weather in {location}: Sunny, 25°C"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)  # Note: eval is unsafe, use a proper parser in production
        return str(result)
    except:
        return "Invalid expression"

def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for '{query}': [Mock results about {query}]"

def test_agent_sync():
    """Test the synchronous Agent class implementation"""
    print("=== Testing Agent (Synchronous) ===\n")
    
    # Initialize the sync vLLM client
    llm = vLLM(
        host="192.168.170.76",
        port="8077",
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        timeout=300
    )
    
    try:
        # Test 1: Basic agent without tools
        print("1. Testing basic sync agent without tools:")
        agent = Agent(llm=llm)
        result = agent("What is 2+2? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Agent with tools using universal __call__
        print("2. Testing sync agent with tools (universal __call__):")
        agent_with_tools = Agent(llm=llm, tools=[get_weather, calculate])
        result = agent_with_tools("What's 5+5 and weather in Paris? /no_think")
        print(f"   Result: {result['content']}")
        print(f"   Iterations: {result['iterations']}\n")
        
        # Test 3: Agent with tools using sync access patterns
        print("3. Testing sync agent with different access patterns:")
        agent_tools = Agent(llm=llm, tools=[calculate, search_web])
        
        # Direct access
        result = agent_tools("Calculate 10*15 /no_think")["content"]
        print(f"   Direct access result: {result}")
        
        # .get() access
        result = agent_tools("Search for Python tutorials /no_think").get("content", "No content")
        print(f"   .get() access result: {result}")
        
        # String conversion
        result = str(agent_tools("What is 7*8? /no_think"))
        print(f"   String conversion result: {result[:100]}...\n")
        
        # Test 4: Agent with streaming
        print("4. Testing sync agent with streaming:")
        stream_agent = Agent(llm=llm, tools=[calculate], stream=True)
        print("   Streaming results:")
        for i, event in enumerate(stream_agent("What is 12*12? /no_think")):
            if event['type'] == 'llm_response':
                print(f"   - LLM Response: {event['content'][:50]}...")
            elif event['type'] == 'tool_start':
                print(f"   - Tool Start: {event['tool_name']}")
            elif event['type'] == 'tool_result':
                print(f"   - Tool Result: {event['result']}")
            elif event['type'] == 'final':
                print(f"   - Final: {event['content']}")
                break
            if i > 10:  # Safety break
                break
        print()
        
        # Test 5: Agent with history
        print("5. Testing sync agent with history:")
        history = [
            {"role": "system", "content": "You are a helpful math assistant"},
            {"role": "user", "content": "I need help with calculations"},
            {"role": "assistant", "content": "I'll help you with math problems!"}
        ]
        agent_with_history = Agent(llm=llm, tools=[calculate], history=history)
        result = agent_with_history("What's 15+25? /no_think")
        print(f"   Result with history: {result['content']}")
        print(f"   Total messages: {len(result['messages'])}\n")
        
        # Test 6: Agent chaining
        print("6. Testing sync agent chaining:")
        search_agent = Agent(llm=llm, tools=[search_web], stream=False)
        calc_agent = Agent(llm=llm, tools=[calculate], stream=False)
        summary_agent = Agent(llm=llm, tools=[], stream=False)
        
        # Chain agents
        chain = search_agent >> calc_agent >> summary_agent
        result = chain("Search Python popularity then calculate 2+2 /no_think")
        print(f"   Chain result: {result['content'][:100]}...")
        print(f"   Chain length: {result['chain_length']}\n")
        
        # Test 7: Agent export and load
        print("7. Testing sync agent export/load:")
        export_agent = Agent(llm=llm, tools=[get_weather])
        
        # Make some conversation first
        export_agent("What's the weather in Tokyo? /no_think")
        
        # Export
        json_data = export_agent.export('json')
        print(f"   Exported JSON (first 100 chars): {json_data[:100]}...")
        
        # Load
        loaded_agent = Agent.load(json_data, llm=llm, tools=[get_weather])
        print(f"   Loaded agent history items: {len(loaded_agent.history)}")
        print(f"   Loaded conversation items: {len(loaded_agent.get_conversation())}\n")
        
        # Test 8: Error handling with sync LLM
        print("8. Testing error handling:")
        try:
            # This should work fine with sync LLM
            simple_agent = Agent(llm=llm)
            result = simple_agent("Hello! /no_think")
            print(f"   Normal sync call works: {result['content'][:50]}...")
        except Exception as e:
            print(f"   Unexpected error: {e}")
        
        print()
        print("✅ All Agent sync tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the sync test
    test_agent_sync()