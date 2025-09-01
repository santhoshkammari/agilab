import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent
import time

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is 22°{unit[0].upper()}, partly cloudy with light breeze."

def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

def test_complete_agent_streaming():
    """Test complete Agent streaming with LLM backend."""
    print("=== Testing Complete Agent Streaming Flow ===")
    
    # Create LLM instance  
    llm = LLM(base_url="http://127.0.0.1:8000")
    
    # Create Agent with tools
    agent = Agent(
        llm=llm,
        tools=[get_weather, calculate_sum],
        stream=True,
        enable_rich_debug=False
    )
    
    print("Testing Agent streaming with weather + math...")
    try:
        stream = agent(
            "What's the weather in London and what's 33 + 17?",
            stream=True,
            max_tokens=300,
            temperature=0.8
        )
        
        print("Stream type:", type(stream))
        
        event_count = 0
        for event in stream:
            event_type = event.get('type', 'unknown')
            
            if event_type == 'iteration_start':
                print(f"\n🔄 ITERATION {event.get('iteration')} START")
            elif event_type == 'token':
                print(f"TOKEN: '{event.get('content', '')}'", end='', flush=True)
            elif event_type == 'think_token':
                print(f"\n💭 THINK TOKEN: '{event.get('content', '')}'", end='', flush=True)
            elif event_type == 'think_block':
                print(f"\n💭 THINK COMPLETE: {event.get('content', '')}")
            elif event_type == 'tool_start':
                print(f"\n🔧 TOOL START: {event.get('tool_name')} with {event.get('tool_args')}")
            elif event_type == 'tool_result':
                print(f"🔧 TOOL RESULT: {event.get('tool_name')} -> {event.get('result')}")
            elif event_type == 'final':
                print(f"\n✅ FINAL: {event.get('content', '')}")
                break
            elif event_type == 'llm_response':
                print(f"\n📦 LLM RESPONSE: content='{event.get('content', '')[:50]}...' tool_calls={len(event.get('tool_calls', []))}")
            
            event_count += 1
            time.sleep(0.02)  # Small delay to see streaming effect
            
            if event_count > 100:  # Limit output
                print("\n... (truncated)")
                break
                
    except Exception as e:
        print(f"Agent streaming error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_agent_streaming()