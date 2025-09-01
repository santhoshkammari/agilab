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

def test_llm_streaming():
    """Test direct LLM streaming behavior."""
    print("=== Testing LLM Streaming ===")
    
    llm = LLM(base_url="http://127.0.0.1:8000")
    
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    
    print("Testing LLM direct streaming...")
    try:
        stream = llm.chat(
            messages, 
            tools=[get_weather, calculate_sum],
            stream=True,
            max_tokens=200,
            temperature=0.8
        )
        
        print("Stream type:", type(stream))
        print("Stream object:", stream)
        
        token_count = 0
        for chunk in stream:
            print(f"Chunk {token_count}: {chunk}")
            token_count += 1
            if token_count > 20:  # Limit output
                print("... (truncated)")
                break
                
    except Exception as e:
        print(f"LLM streaming error: {e}")
    
    print("\n")

def test_agent_streaming():
    """Test Agent streaming behavior."""
    print("=== Testing Agent Streaming ===")
    
    llm = LLM(base_url="http://127.0.0.1:8000")
    agent = Agent(
        llm=llm,
        tools=[get_weather, calculate_sum],
        stream=True,
        enable_rich_debug=False
    )
    
    print("Testing Agent streaming...")
    try:
        stream = agent("What's the weather in Paris?", stream=True, max_tokens=200)
        
        print("Stream type:", type(stream))
        
        event_count = 0
        for event in stream:
            print(f"Event {event_count}: {event}")
            event_count += 1
            time.sleep(0.1)  # Small delay to see streaming effect
            if event_count > 15:  # Limit output
                print("... (truncated)")
                break
                
    except Exception as e:
        print(f"Agent streaming error: {e}")

def test_llm_non_streaming():
    """Test non-streaming for comparison."""
    print("=== Testing LLM Non-Streaming (for comparison) ===")
    
    llm = LLM(base_url="http://127.0.0.1:8000")
    
    result = llm.chat(
        [{"role": "user", "content": "What's 5 + 7?"}],
        tools=[get_weather, calculate_sum],
        stream=False,
        max_tokens=200
    )
    
    print("Non-streaming result:", result)

if __name__ == "__main__":
    print("Testing streaming behavior...\n")
    
    test_llm_non_streaming()
    print("\n" + "="*50 + "\n")
    
    test_llm_streaming()
    print("\n" + "="*50 + "\n")
    
    test_agent_streaming()