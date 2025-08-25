from .basellm import GeminiLLM

def validate_messages(messages):
    """Test if messages is a list of dict with role, content keys"""
    if not isinstance(messages, list):
        return False
    
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if not all(key in msg for key in ['role', 'content']):
            return False
    
    return True

if __name__ == '__main__':

    # Test with dict format
    test_messages = [
        {"role": "system", "content": "You are a pirate with a colorful personality"},
        {"role": "user", "content": "Tell me a story"}
    ]

    print(f"Message validation test: {validate_messages(test_messages)}")

    # Test the Gemini class
    llm = GeminiLLM(model="gemini-2.0-flash")

    # Test with __call__ method (single string)
    print("\n=== Testing with single string ===")
    resp1 = llm("Tell me a joke about pirates")
    print(f"Response: {resp1}")

    # Test with __call__ method (list of messages)
    print("\n=== Testing with message list ===")
    resp2 = llm(test_messages)
    print(f"Response: {resp2}")

    # Test with chat method directly
    print("\n=== Testing chat method directly ===")
    resp3 = llm.chat(test_messages)
    print(f"Response: {resp3}")

    # Test streaming
    print("\n=== Testing streaming ===")
    from llama_index.core.llms import ChatMessage

    stream_messages = [
        ChatMessage(role="user", content="Tell me a short joke about programming"),
    ]

    gemini_llm = llm.llm  # Get the underlying GoogleGenAI instance
    resp_stream = gemini_llm.stream_chat(stream_messages)

    print("Streaming response:")
    full_response = ""
    for r in resp_stream:
        print(r.delta, end="")
        full_response += r.delta
    print(f"\n\nFull streamed response: {full_response}")

    # Test streaming with Gemini class
    print("\n=== Testing Gemini streaming ===")
    stream_resp = llm.stream_chat("Tell me a very short programming joke")

    print("Gemini streaming response:")
    full_gemini_response = ""
    for chunk in stream_resp:
        print(chunk["delta"], end="")
        full_gemini_response += chunk["delta"]
    print(f"\n\nFull Gemini streamed response: {full_gemini_response}")

    # Test streaming via __call__ method
    print("\n=== Testing Gemini __call__ with stream=True ===")
    stream_resp2 = llm("Why do developers hate debugging?", stream=True)

    print("__call__ streaming response:")
    full_call_response = ""
    for chunk in stream_resp2:
        print(chunk["delta"], end="")
        full_call_response += chunk["delta"]
    print(f"\n\nFull __call__ streamed response: {full_call_response}")