#!/usr/bin/env python3
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import vLLM

def test_vllm_sync():
    """Test the synchronous vLLM class implementation"""
    print("=== Testing vLLM (Synchronous) ===\n")
    
    # Initialize the sync vLLM client with the same config as the user provided
    llm = vLLM(
        host="192.168.170.76",
        port="8077",
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        timeout=300  # 5 minutes timeout for CPU operations
    )
    
    try:
        # Test 1: Basic sync chat
        print("1. Testing basic sync chat:")
        result = llm.chat("What is 2+2? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Using __call__ method synchronously
        print("2. Testing sync __call__ method:")
        result = llm("What is 3+3? /no_think")
        print(f"   Result: {result['content']}\n")
        
        # Test 3: Sync batch processing
        print("3. Testing sync batch processing:")
        batch_inputs = [
            "What is 5+5? /no_think",
            "What is 7+7? /no_think",
            "What is 9+9? /no_think"
        ]
        batch_results = llm.batch_call(batch_inputs, max_workers=3)
        print(f"   Batch results: {[r['content'] for r in batch_results]}\n")
        
        # Test 4: Sync with conversation format
        print("4. Testing sync conversation:")
        conversation = [
            {"role": "user", "content": "Hello! Just say hi back. /no_think"}
        ]
        result = llm(conversation)
        print(f"   Conversation result: {result['content']}\n")
        
        # Test 5: Auto-batching with list of strings
        print("5. Testing auto-batching with list of strings:")
        simple_batch = ["What is 10+10? /no_think", "What is 20+20? /no_think"]
        auto_batch_results = llm(simple_batch)
        print(f"   Auto-batch results: {[r['content'] for r in auto_batch_results]}\n")
        
        print("✅ All vLLM sync tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the sync test
    test_vllm_sync()