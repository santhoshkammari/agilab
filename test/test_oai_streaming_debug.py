#!/usr/bin/env python3

import sys
import os
import json

# Add the claude directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude', 'llm'))
from __oai import OAI

def test_oai_streaming_debug():
    """Debug OAI streaming response format."""
    
    print("🔍 Testing OAI streaming with Google Gemini...")
    print("=" * 60)
    
    # Initialize OAI client with OpenRouter + Gemini
    oai = OAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    print("Configuration:")
    print(f"  Base URL: https://openrouter.ai/api/v1")
    print(f"  Model: google/gemini-2.0-flash-exp:free")
    print()
    
    try:
        print("🚀 Starting streaming request...")
        chunk_count = 0
        
        for chunk in oai(
            prompt="hi",
            model="google/gemini-2.0-flash-exp:free",
            temperature=0.7,
            stream=True
        ):
            chunk_count += 1
            print(f"\n📦 Chunk {chunk_count}:")
            print(f"Type: {type(chunk)}")
            print(f"Content: {chunk}")
            
            # Check for specific attributes
            if isinstance(chunk, dict):
                print(f"Keys: {list(chunk.keys())}")
                
                if "error" in chunk:
                    print(f"❌ Error in chunk: {chunk['error']}")
                    break
                    
                if "message" in chunk:
                    print(f"📄 Message: {chunk['message']}")
                    if isinstance(chunk['message'], dict):
                        print(f"  Message keys: {list(chunk['message'].keys())}")
                        if 'content' in chunk['message']:
                            print(f"  Content: {chunk['message']['content']}")
                
                if chunk.get("done", False):
                    print(f"✅ Stream finished: {chunk.get('done_reason', 'completed')}")
                    break
            else:
                print(f"⚠️  Unexpected chunk type: {type(chunk)}")
            
            # Limit output for debugging
            if chunk_count > 20:
                print("🛑 Stopping after 20 chunks for debugging...")
                break
        
        print(f"\n📊 Total chunks processed: {chunk_count}")
        
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🔍 Debug test complete!")

if __name__ == "__main__":
    test_oai_streaming_debug()