#!/usr/bin/env python3
"""
Test vLLM streaming directly with OpenAI client.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openai import OpenAI


def test_vllm_streaming():
    """Test vLLM streaming directly."""
    print("=== Testing vLLM Streaming Directly ===")
    
    # Initialize OpenAI client for vLLM
    client = OpenAI(
        api_key="dummy",
        base_url="http://192.168.170.76:8077/v1"
    )
    
    print("1. Testing non-streaming call...")
    try:
        response = client.chat.completions.create(
            model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            messages=[{"role": "user", "content": "what is 2+3?"}],
            stream=False
        )
        print(f"Non-streaming response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Non-streaming error: {str(e)}")
        return
    
    print("\n2. Testing streaming call...")
    try:
        stream = client.chat.completions.create(
            model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            messages=[{"role": "user", "content": "what is 2+3?"}],
            stream=True
        )
        
        print("Streaming chunks:")
        content_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_parts.append(content)
                print(f"Chunk: {repr(content)}")
        
        print(f"\nFull content: {''.join(content_parts)}")
        
    except Exception as e:
        print(f"Streaming error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vllm_streaming()