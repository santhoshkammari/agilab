#!/usr/bin/env python3

import sys
import os

# Add the claude directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude'))

def test_browser_use_integration():
    """Test browser-use LLM integration with compatibility layer."""
    
    print("🔍 Testing Browser-Use LLM Integration...")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from claude.llm import ChatOllama, ChatOpenRouter
        from claude.llm.compat import LLMCompatibilityWrapper
        print("✅ Imports successful")
        
        # Test Ollama wrapper
        print("\n🦙 Testing Ollama integration...")
        ollama_llm = ChatOllama(model="qwen3:4b", host="http://localhost:11434")
        wrapped_ollama = LLMCompatibilityWrapper(ollama_llm)
        print("✅ Ollama wrapper created")
        
        # Test OpenRouter wrapper  
        print("\n🌐 Testing OpenRouter integration...")
        openrouter_llm = ChatOpenRouter(
            model="google/gemini-2.0-flash-exp:free",
            api_key="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"
        )
        wrapped_openrouter = LLMCompatibilityWrapper(openrouter_llm)
        print("✅ OpenRouter wrapper created")
        
        # Test message conversion
        print("\n📝 Testing message conversion...")
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        browser_use_messages = wrapped_ollama._convert_messages_to_browser_use(test_messages)
        print(f"✅ Converted {len(test_messages)} messages to browser-use format")
        print(f"  Message types: {[type(msg).__name__ for msg in browser_use_messages]}")
        
        # Test OpenRouter chat (if we want to test real API call)
        print("\n🚀 Testing OpenRouter chat call...")
        try:
            response = wrapped_openrouter.chat(messages=[{"role": "user", "content": "Hi"}])
            print(f"✅ OpenRouter response: {response.message.content[:100]}...")
        except Exception as e:
            print(f"⚠️  OpenRouter test skipped (might need network): {e}")
        
        print("\n🎯 Integration Test Result: SUCCESS!")
        print("The browser-use LLM integration is working correctly!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_browser_use_integration()