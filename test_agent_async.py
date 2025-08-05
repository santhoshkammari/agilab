#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import vLLMAsync, vLLM
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

async def async_search_web(query: str) -> str:
    """Async search the web for information"""
    await asyncio.sleep(0.1)  # Simulate async operation
    return f"Async search results for '{query}': [Mock async results about {query}]"

def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for '{query}': [Mock results about {query}]"

async def test_agent_async():
    """Test the asynchronous Agent class implementation"""
    print("=== Testing Agent (Asynchronous) ===\n")
    
    # Initialize both sync and async vLLM clients for testing
    async_llm = vLLMAsync(
        host="192.168.170.76",
        port="8077",
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        timeout=300
    )
    
    sync_llm = vLLM(
        host="192.168.170.76",
        port="8077", 
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        timeout=300
    )
    
    try:
        # Test 1: Basic async agent with async LLM
        print("1. Testing async agent with async LLM:")
        agent = Agent(llm=async_llm)
        result = await agent("What is 2+2? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Async agent with async LLM and tools
        print("2. Testing async agent with async LLM and tools:")
        agent_with_tools = Agent(llm=async_llm, tools=[get_weather, calculate, async_search_web])
        result = await agent_with_tools("What's 5+5 and weather in London? /no_think")
        print(f"   Result: {result['content']}")
        print(f"   Iterations: {result['iterations']}\n")
        
        # Test 3: Sync LLM in async context using await
        print("3. Testing sync LLM in async context with await:")
        sync_agent = Agent(llm=sync_llm, tools=[calculate])
        result = await sync_agent("Calculate 10*15 /no_think")
        print(f"   Sync LLM in async context result: {result['content']}\n")
        
        # Test 4: Universal __call__ with async LLM (await required)
        print("4. Testing universal __call__ with async LLM:")
        universal_agent = Agent(llm=async_llm, tools=[get_weather])
        try:
            # This should work - awaiting the call
            result = await universal_agent("What's the weather in Tokyo? /no_think")
            print(f"   Awaited call result: {result['content']}")
        except Exception as e:
            print(f"   Expected behavior for async LLM: {e}")
        
        # Direct access should fail with async LLM
        try:
            result = universal_agent("What's the weather in Paris? /no_think")["content"]
            print(f"   Direct access result: {result}")
        except RuntimeError as e:
            print(f"   Expected error for direct access with async LLM: {e}")
        print()
        
        # Test 5: Async agent with streaming
        print("5. Testing async agent with streaming:")
        stream_agent = Agent(llm=async_llm, tools=[calculate], stream=True)
        print("   Async streaming results:")
        i = 0
        async for event in await stream_agent("What is 12*12? /no_think"):
            if event['type'] == 'llm_response':
                print(f"   - LLM Response: {event['content'][:50]}...")
            elif event['type'] == 'tool_start':
                print(f"   - Tool Start: {event['tool_name']}")
            elif event['type'] == 'tool_result':
                print(f"   - Tool Result: {event['result']}")
            elif event['type'] == 'final':
                print(f"   - Final: {event['content']}")
                break
            i += 1
            if i > 10:  # Safety break
                break
        print()
        
        # Test 6: Async agent with history
        print("6. Testing async agent with history:")
        history = [
            {"role": "system", "content": "You are a helpful math assistant"},
            {"role": "user", "content": "I need help with calculations"},
            {"role": "assistant", "content": "I'll help you with math problems!"}
        ]
        agent_with_history = Agent(llm=async_llm, tools=[calculate], history=history)
        result = await agent_with_history("What's 15+25? /no_think")
        print(f"   Result with history: {result['content']}")
        print(f"   Total messages: {len(result['messages'])}\n")
        
        # Test 7: Explicit acall() method
        print("7. Testing explicit acall() method:")
        # With async LLM
        async_agent = Agent(llm=async_llm, tools=[get_weather])
        result = await async_agent.acall("What's the weather in Berlin? /no_think")
        print(f"   acall() with async LLM: {result['content']}")
        
        # With sync LLM
        sync_agent = Agent(llm=sync_llm, tools=[calculate])
        result = await sync_agent.acall("What is 7*8? /no_think")
        print(f"   acall() with sync LLM: {result['content']}\n")
        
        # Test 8: Async batch operations
        print("8. Testing async batch operations:")
        batch_agent = Agent(llm=async_llm, tools=[calculate], stream=False)
        
        # Create multiple async calls
        tasks = [
            batch_agent("What is 2*3? /no_think"),
            batch_agent("What is 4*5? /no_think"),
            batch_agent("What is 6*7? /no_think")
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        print(f"   Async batch results: {[r['content'] for r in results]}\n")
        
        # Test 9: Mixed sync/async tools
        print("9. Testing mixed sync/async tools:")
        mixed_agent = Agent(llm=async_llm, tools=[calculate, async_search_web, get_weather], stream=False)
        result = await mixed_agent("Calculate 3*4 then search for Python and get weather in NYC /no_think")
        print(f"   Mixed tools result: {result['content']}")
        print(f"   Iterations: {result['iterations']}\n")
        
        # Test 10: Agent export in async context
        print("10. Testing agent export in async context:")
        export_agent = Agent(llm=async_llm, tools=[get_weather])
        
        # Make some conversation first
        await export_agent("What's the weather in Sydney? /no_think")
        
        # Export
        json_data = export_agent.export('json')
        print(f"   Exported JSON (first 100 chars): {json_data[:100]}...")
        
        # Load
        loaded_agent = Agent.load(json_data, llm=async_llm, tools=[get_weather])
        result = await loaded_agent("What's the weather in Melbourne? /no_think")
        print(f"   Loaded agent works: {result['content'][:50]}...\n")
        
        # Test 11: Error handling with async LLM
        print("11. Testing error handling with async LLM:")
        error_agent = Agent(llm=async_llm)
        
        # This should work
        result = await error_agent("Hello! /no_think")
        print(f"   Async call works: {result['content'][:50]}...")
        
        # This should fail gracefully
        try:
            # Direct access with async LLM should fail
            result = error_agent("Hello!")["content"]
            print(f"   Unexpected success: {result}")
        except RuntimeError as e:
            print(f"   Expected error for direct access: {e}")
        
        print()
        
        # Test 12: Performance comparison
        print("12. Testing performance comparison:")
        
        # Sync agent timing
        import time
        sync_agent = Agent(llm=sync_llm, tools=[calculate], stream=False)
        start_time = time.time()
        await sync_agent.acall("What is 100*100? /no_think")
        sync_time = time.time() - start_time
        
        # Async agent timing  
        async_agent = Agent(llm=async_llm, tools=[calculate], stream=False)
        start_time = time.time()
        await async_agent("What is 100*100? /no_think")
        async_time = time.time() - start_time
        
        print(f"   Sync LLM in async context: {sync_time:.3f}s")
        print(f"   Async LLM: {async_time:.3f}s\n")
        
        print("✅ All Agent async tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_agent_async())