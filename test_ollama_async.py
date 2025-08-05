#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import OllamaAsync

async def test_ollama_async():
    """Test the OllamaAsync class implementation"""
    print("=== Testing OllamaAsync ===\n")
    
    # Initialize the async Ollama client
    # You can configure these parameters as needed for your Ollama setup
    llm = OllamaAsync(
        host="192.168.170.76",  # or "192.168.170.76" if remote  
        port="11434",     # default Ollama port
        model="qwen3:0.6b-fp16",  # or whatever model you have available
        timeout=30
    )
    
    try:
        # Test 1: Basic async chat
        print("1. Testing basic async chat:")
        result = await llm.chat("What is 2+2?")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Using async __call__ method
        print("2. Testing async __call__ method:")
        result = await llm("What is 3+3?")
        print(f"   Result: {result['content']}\n")
        
        # Test 3: Async batch processing
        print("3. Testing async batch processing:")
        batch_inputs = [
            "What is 5+5?",
            "What is 7+7?", 
            "What is 9+9?"
        ]
        batch_results = await llm.batch_call(batch_inputs, max_workers=3)
        print(f"   Batch results: {[r['content'] for r in batch_results]}\n")
        
        # Test 4: Async with conversation format
        print("4. Testing async conversation:")
        conversation = [
            {"role": "user", "content": "Hello! Just say hi back."}
        ]
        result = await llm(conversation)
        print(f"   Conversation result: {result['content']}\n")
        
        # Test 5: Auto-batching with async processing
        print("5. Testing async auto-batching with list of strings:")
        simple_batch = ["What is 10+10?", "What is 20+20?"]
        auto_batch_results = await llm(simple_batch)
        print(f"   Auto-batch results: {[r['content'] for r in auto_batch_results]}\n")
        
        # Test 6: Async thinking extraction (if supported by model)
        print("6. Testing async thinking extraction:")
        result = await llm("Explain step by step: What is 2+3?")
        print(f"   Content: {result['content']}")
        print(f"   Thinking: {result['think']}\n")
        
        # Test 7: Concurrent async requests
        print("7. Testing concurrent async requests:")
        tasks = [
            llm("What is 1+1?"),
            llm("What is 2+2?"),
            llm("What is 3+3?")
        ]
        concurrent_results = await asyncio.gather(*tasks)
        print(f"   Concurrent results: {[r['content'] for r in concurrent_results]}\n")
        
        # Test 8: Mixed conversation types in batch
        print("8. Testing mixed batch with conversations:")
        mixed_batch = [
            [{"role": "user", "content": "Say hello"}],
            [{"role": "user", "content": "Say goodbye"}]
        ]
        mixed_results = await llm(mixed_batch)
        print(f"   Mixed batch results: {[r['content'] for r in mixed_results]}\n")
        
        print("✅ All OllamaAsync tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Note: Make sure Ollama is running and the specified model is available")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_ollama_async())