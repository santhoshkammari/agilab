#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import vLLMAsync

async def test_vllm_async():
    """Test the vLLMAsync class implementation"""
    print("=== Testing vLLMAsync ===\n")
    
    # Initialize the async vLLM client with the same config as the user provided
    llm = vLLMAsync(
        host="192.168.170.76",
        port="8077",
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        timeout=300  # 5 minutes timeout for CPU operations
    )
    
    try:
        # Test 1: Basic async chat
        print("1. Testing basic async chat:")
        result = await llm.chat("What is 2+2? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Using __call__ method asynchronously
        print("2. Testing async __call__ method:")
        result = await llm("What is 3+3? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 3: Async batch processing
        print("3. Testing async batch processing:")
        batch_inputs = [
            "What is 5+5? /no_think",
            "What is 7+7? /no_think",
            "What is 9+9? /no_think"
        ]
        batch_results = await llm.batch_call(batch_inputs, max_workers=3)
        print(f"   Batch results: {[r['content'] for r in batch_results]}\n")
        
        # Test 4: Async with conversation format
        print("4. Testing async conversation:")
        conversation = [
            {"role": "user", "content": "Hello! Just say hi back. /no_think"}
        ]
        result = await llm(conversation)
        print(f"   Conversation result: {result['content']}\n")
        
        print("✅ All vLLMAsync tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_vllm_async())