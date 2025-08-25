from flowgen.llm import Ollama,LlamaCpp
from flowgen.llm.basellm import BaseLLM, Restaurant, FriendList, get_weather, get_current_time, add_two_numbers

if __name__ == "__main__":
    # Initialize Ollama
    # NOTE: Make sure Ollama is running with: ollama serve
    # llm = Ollama(model="smollm2:135m")
    llm = LlamaCpp(model='/home/ntlpt59/Downloads/LFM2-350M-F16.gguf')
    print("=== Testing basic chat ===")
    try:
        response = llm("Tell me a short joke about programming")
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in basic chat: {e}")
        print("Make sure Ollama is running: ollama serve")

    # Test tools with Python functions - now automatic!
    print("\n=== Testing tools automatically ===")
    try:
        response = llm("What's the weather like in New York?", tools=[get_weather])
        print(f"Tool response: {response}")
        if response.get('tool_calls'):
            print("Tool calls detected - would execute functions automatically")
    except Exception as e:
        print(f"Error in tool calling: {e}")
        print("Make sure you have a model that supports function calling")
    # Test math tools
    print("\n=== Testing math tools ===")
    try:
        response = llm("What is 25 plus 17?", tools=[add_two_numbers])
        print(f"Math response: {response}")
        if response.get('tool_calls'):
            print("Math tool calls detected")
    except Exception as e:
        print(f"Error in math tools: {e}")
#
#     # Test structured output
#     print("\n=== Testing structured output ===")
#     try:
#         response = llm("Generate a restaurant in Miami", format=Restaurant)
#         print(f"Structured response: {response['content']}")
#     except Exception as e:
#         print(f"Error in structured output: {e}")
#
#     # Test friends list structured output
#     print("\n=== Testing friends list ===")
#     try:
#         response = llm("I have two friends. Alice is 25 and available, Bob is 30 and busy", format=FriendList)
#         print(f"Friends response: {response['content']}")
#     except Exception as e:
#         print(f"Error in friends list: {e}")
#
#     # Test streaming
#     print("\n=== Testing streaming ===")
#     try:
#         stream_response = llm("Tell me a short story about AI", stream=True)
#         print("Streaming response:")
#         if isinstance(stream_response, dict):
#             print(stream_response['content'])
#         else:
#             for chunk in stream_response:
#                 print(chunk["content"], end="", flush=True)
#         print()
#     except Exception as e:
#         print(f"Error in streaming: {e}")
#
#     # Test with message history
#     print("\n=== Testing message history ===")
#     try:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": "What is the capital of France?"}
#         ]
#         response = llm(messages)
#         print(f"Response: {response['content']}")
#     except Exception as e:
#         print(f"Error in message history: {e}")
#
#     print("\n=== Simple Usage Examples ===")
#     print("llm('Hello')  # Basic chat")
#     print("llm('Generate person', format=PersonSchema)  # Structured output")
#     print("llm('What weather?', tools=[weather_func])  # Function calling")
#     print("llm(messages)  # Multi-turn chat")
#     print("llm(text, stream=True)  # Streaming")
#
#     print("\nTo use Ollama:")
#     print("1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
#     print("2. Start: ollama serve")
#     print("3. Pull model: ollama pull llama3.1")
#     print("4. Run this script!")


from llama_cpp import Llama
llm = Llama(model_path="/home/ntlpt59/Downloads/LFM2-350M-F16.gguf")
res = llm.create_chat_completion(
      messages = [
        {
          "role": "system",
          "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

        },
        {
          "role": "user",
          "content": "Extract Jason is 25 years old"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
      tool_choice={
        "type": "function",
        "function": {
          "name": "UserDetail"
        }
      }
)

print(res)