#!/usr/bin/env python3
"""
Test script for hfLLM Transformers LLM implementation
"""

import asyncio
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, '/home/ntlpt59/master/own/flowgen')

from flowgen.llm.llm import hgLLM, hgLLMAsync

def test_sync_transformers():
    """Test synchronous hfLLM Transformers LLM"""
    print("=== Testing Sync hfLLM Transformers LLM ===")
    
    try:
        # Initialize with a small model (you can change this to any model you have)
        llm = hgLLM(
            model="Qwen/Qwen3-1.7B",  # Using the model from user's example
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Test basic text generation
        print("1. Basic text generation:")
        result = llm("What is 2+2?", max_new_tokens=50)
        print(f"Content: {result['content']}")
        print(f"Think: {result['think']}")
        print(f"Tool calls: {result['tool_calls']}")
        print()
        
        # Test with conversation format
        print("2. Conversation format:")
        messages = [
            {"role": "user", "content": "Hello! What's your name?"}
        ]
        result = llm(messages, max_new_tokens=50)
        print(f"Content: {result['content']}")
        print()
        
        # Test streaming
        print("3. Streaming generation:")
        result = llm("Tell me a short joke", stream=True, max_new_tokens=100)
        print(f"Content: {result['content']}")
        print()
        
        print("✅ Sync tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        return False

async def test_async_transformers():
    """Test asynchronous hfLLM Transformers LLM"""
    print("=== Testing Async hfLLM Transformers LLM ===")
    
    try:
        # Initialize with a small model
        llm = hgLLMAsync(
            model="Qwen/Qwen3-1.7B",  # Using the model from user's example
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Test basic text generation
        print("1. Basic async text generation:")
        result = await llm("What is 3+3?", max_new_tokens=50)
        print(f"Content: {result['content']}")
        print(f"Think: {result['think']}")
        print(f"Tool calls: {result['tool_calls']}")
        print()
        
        # Test batch processing
        print("2. Async batch processing:")
        texts = ["What is 1+1?", "What is 5+5?"]
        results = await llm(texts, max_new_tokens=30)
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['content']}")
        print()
        
        print("✅ Async tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing hfLLM Transformers LLM Implementation")
    print("=" * 60)
    
    # Test sync version
    sync_success = test_sync_transformers()
    print()
    
    # Test async version
    async_success = asyncio.run(test_async_transformers())
    
    print("\n" + "=" * 60)
    if sync_success and async_success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        
    print("\nNote: Make sure you have the transformers library installed:")
    print("pip install transformers torch")
    print("\nAnd ensure the model 'Qwen/Qwen3-1.7B' is available or change to a model you have access to.")

if __name__ == "__main__":
    main()