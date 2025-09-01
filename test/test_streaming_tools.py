import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flowgen.llm.llm import LLM

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is 22°{unit[0].upper()}, partly cloudy with light breeze."

def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

def test_streaming_tool_calls():
    """Test streaming tool calls with weather and sum functions."""
    
    # Initialize LLM client
    llm = LLM(base_url="http://127.0.0.1:8000")
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate_sum",
                "description": "Calculate the sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number", 
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    
    print("=== Testing Weather Query with Streaming ===")
    
    # Test 1: Weather query
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]
    
    print("Requesting weather information...")
    stream = llm.chat(messages, tools=tools, stream=True, max_tokens=200)
    
    full_content = ""
    tool_calls_received = []
    
    for chunk in stream:
        if chunk["content"]:
            print(f"Content chunk: {chunk['content']}")
            full_content += chunk["content"]
        
        if chunk["tool_calls"]:
            print(f"Tool calls received: {chunk['tool_calls']}")
            tool_calls_received.extend(chunk["tool_calls"])
    
    print(f"Final content: {full_content}")
    print(f"Tool calls: {tool_calls_received}")
    
    # Execute tool calls if any
    if tool_calls_received:
        for tool_call in tool_calls_received:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            if func_name == "get_weather":
                result = get_weather(**args)
                print(f"Weather result: {result}")
    
    print("\n=== Testing Sum Calculation with Streaming ===")
    
    # Test 2: Sum calculation
    messages2 = [
        {"role": "user", "content": "What is 15 + 27?"}
    ]
    
    print("Requesting sum calculation...")
    stream2 = llm.chat(messages2, tools=tools, stream=True, max_tokens=200)
    
    full_content2 = ""
    tool_calls_received2 = []
    
    for chunk in stream2:
        if chunk["content"]:
            print(f"Content chunk: {chunk['content']}")
            full_content2 += chunk["content"]
        
        if chunk["tool_calls"]:
            print(f"Tool calls received: {chunk['tool_calls']}")
            tool_calls_received2.extend(chunk["tool_calls"])
    
    print(f"Final content: {full_content2}")
    print(f"Tool calls: {tool_calls_received2}")
    
    # Execute tool calls if any
    if tool_calls_received2:
        for tool_call in tool_calls_received2:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            if func_name == "calculate_sum":
                result = calculate_sum(**args)
                print(f"Sum result: {result}")

if __name__ == "__main__":
    test_streaming_tool_calls()