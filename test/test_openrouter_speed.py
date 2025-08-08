#!/usr/bin/env python3
"""
Test OpenRouter speed directly to diagnose slowness issues
"""
import time
import asyncio
from llama_index.llms.openrouter import OpenRouter

async def test_openrouter_speed():
    """Test OpenRouter API response time"""
    
    print("🧪 Testing OpenRouter Speed...")
    print("=" * 50)
    
    # Configuration from your code
    api_key = "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"
    model = "qwen/qwen3-coder:free"
    
    # Test with increased limits
    llm = OpenRouter(
        api_key=api_key,
        max_tokens=2048,  # Increased from 256
        context_window=8096,  # Using your config value
        model=model,
        timeout=30.0  # Add explicit timeout
    )
    
    print(f"Model: {model}")
    print(f"Max tokens: 2048")
    print(f"Context window: 8096")
    print()
    
    # Test simple completion
    test_messages = [
        "Hi, how are you?",
        "What is 2+2?", 
        "Write a simple hello world in Python"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"📝 Test {i}: {message}")
        
        start_time = time.time()
        try:
            response = await llm.acomplete(message)
            end_time = time.time()
            
            response_time = end_time - start_time
            print(f"✅ Response time: {response_time:.2f} seconds")
            print(f"📄 Response: {response.text[:100]}...")
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Error after {response_time:.2f} seconds: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_openrouter_speed())