#!/usr/bin/env python3
import sys
import os

# Add the flowgen package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flowgen'))

from flowgen.llm.llm import Ollama

def test_ollama_sync():
    """Test the synchronous Ollama class implementation"""
    print("=== Testing Ollama (Synchronous) ===\n")
    
    # Initialize the sync Ollama client 
    # You can configure these parameters as needed for your Ollama setup
    llm = Ollama(
        host="192.168.170.76",  # or "192.168.170.76" if remote
        port="11434",     # default Ollama port
        model="qwen3:0.6b-fp16",  # or whatever model you have available
        timeout=30
    )
    
    try:
        # Test 1: Basic sync chat
        print("1. Testing basic sync chat:")
        result = llm.chat("What is 2+2?")
        print(f"   Result: {result['content']}\n")
        
        # Test 2: Using __call__ method synchronously
        print("2. Testing sync __call__ method:")
        result = llm("What is 3+3?")
        print(f"   Result: {result['content']}\n")
        
        # Test 3: Sync batch processing
        print("3. Testing sync batch processing:")
        batch_inputs = [
            "What is 5+5?",
            "What is 7+7?",
            "What is 9+9?"
        ]
        batch_results = llm.batch_call(batch_inputs, max_workers=3)
        print(f"   Batch results: {[r['content'] for r in batch_results]}\n")
        
        # Test 4: Sync with conversation format
        print("4. Testing sync conversation:")
        conversation = [
            {"role": "user", "content": "Hello! Just say hi back."}
        ]
        result = llm(conversation)
        print(f"   Conversation result: {result['content']}\n")
        
        # Test 5: Auto-batching with list of strings
        print("5. Testing auto-batching with list of strings:")
        simple_batch = ["What is 10+10?", "What is 20+20?"]
        auto_batch_results = llm(simple_batch)
        print(f"   Auto-batch results: {[r['content'] for r in auto_batch_results]}\n")
        
        # Test 6: Thinking extraction (if supported by model)
        print("6. Testing thinking extraction:")
        result = llm("Explain step by step: What is 2+3?")
        print(f"   Content: {result['content']}")
        print(f"   Thinking: {result['think']}\n")
        
        print("✅ All Ollama sync tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Note: Make sure Ollama is running and the specified model is available")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the sync test
    test_ollama_sync()