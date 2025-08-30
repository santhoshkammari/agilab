from flowgen.llm import LLM
from flowgen.llm.basellm import get_weather, add_two_numbers, Restaurant

def test_basic_usage():
    """Test basic LLM usage with string input."""
    print("=== Testing Basic Usage ===")
    llm = LLM()
    
    try:
        response = llm("hai")
        print(f"Response: {response['content']}")
        print(f"Tool calls: {response['tool_calls']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure API server is running on http://0.0.0.0:8000")

def test_with_tools():
    """Test LLM with tool calling."""
    print("\n=== Testing Tool Calling ===")
    llm = LLM()
    
    try:
        response = llm("What's the weather in Tokyo?", tools=[get_weather])
        print(f"Response: {response['content']}")
        print(f"Tool calls: {response['tool_calls']}")
    except Exception as e:
        print(f"Error in tool calling: {e}")

def test_math_tools():
    """Test LLM with math tools."""
    print("\n=== Testing Math Tools ===")
    llm = LLM()
    
    try:
        response = llm("What is 15 plus 25?", tools=[add_two_numbers])
        print(f"Response: {response['content']}")
        print(f"Tool calls: {response['tool_calls']}")
    except Exception as e:
        print(f"Error in math tools: {e}")

def test_structured_output():
    """Test LLM with structured output."""
    print("\n=== Testing Structured Output ===")
    llm = LLM()
    
    try:
        response = llm("Generate a restaurant in Miami", format=Restaurant)
        print(f"Structured response: {response['content']}")
    except Exception as e:
        print(f"Error in structured output: {e}")

def test_message_history():
    """Test LLM with message history."""
    print("\n=== Testing Message History ===")
    llm = LLM()
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = llm(messages)
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in message history: {e}")

def test_streaming():
    """Test LLM with streaming."""
    print("\n=== Testing Streaming ===")
    llm = LLM()
    
    try:
        stream_response = llm("Tell me a short story about AI", stream=True)
        print("Streaming response:")
        if hasattr(stream_response, '__iter__'):
            for chunk in stream_response:
                if chunk.get("content"):
                    print(chunk["content"], end="", flush=True)
        print()
    except Exception as e:
        print(f"Error in streaming: {e}")

if __name__ == "__main__":
    print("FlowGen LLM HTTP Client Test")
    print("=" * 40)
    print("Make sure your API server is running on http://0.0.0.0:8000")
    print("Start with: python flowgen/api/api.py")
    print()
    
    # Run all tests
    test_basic_usage()
    test_with_tools()
    test_math_tools()
    test_structured_output()
    test_message_history()
    test_streaming()
    
    print("\n=== Usage Examples ===")
    print("llm = LLM()                                    # Basic initialization")
    print("llm('hai')                                     # Simple chat")
    print("llm('hello', tools=[get_weather])              # With tools")
    print("llm('generate data', format=MySchema)          # Structured output")
    print("llm(messages)                                  # Multi-turn chat")
    print("llm('story', stream=True)                      # Streaming")