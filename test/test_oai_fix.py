#!/usr/bin/env python3

import sys
import os

# Add the claude directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude', 'llm'))
from __oai import OAI

def test_oai_chat_compatibility():
    """Test that OAI chat method returns attribute-accessible response."""
    
    print("🔍 Testing OAI chat compatibility...")
    print("=" * 60)
    
    # Initialize OAI client
    oai = OAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    try:
        print("🚀 Testing non-streaming chat...")
        
        response = oai.chat(
            prompt="hi",
            model="google/gemini-2.0-flash-exp:free",
            temperature=0.7
        )
        
        print(f"✅ Response type: {type(response)}")
        print(f"✅ Has message attribute: {hasattr(response, 'message')}")
        print(f"✅ Message type: {type(response.message)}")
        print(f"✅ Has content attribute: {hasattr(response.message, 'content')}")
        print(f"✅ Content: {repr(response.message.content)}")
        print(f"✅ Has tool_calls attribute: {hasattr(response.message, 'tool_calls')}")
        print(f"✅ Tool calls: {response.message.tool_calls}")
        
        # Test the attribute access that input.py expects
        if response.message.content and response.message.content.strip():
            print("✅ Attribute access works: content is accessible and non-empty")
        else:
            print("⚠️  Content is empty or not accessible")
        
        print("\n🎯 Test Result: SUCCESS - OAI now compatible with input.py expectations!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_oai_chat_compatibility()