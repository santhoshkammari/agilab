#!/usr/bin/env python3
"""
Test Google provider with fixed tool schemas
"""

import json
from claude.llm import OAI
from claude.tools import get_tool_schemas

def test_google_provider_with_tools():
    """Test Google provider with tool schemas"""
    print("=== Testing Google Provider with Tool Schemas ===")
    
    try:
        # Get tool schemas
        tools = get_tool_schemas()
        print(f"Generated {len(tools)} tool schemas")
        
        # Create Google LLM client
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",  # From config
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "What tools do you have access to?"}
        ]
        
        print("Making request to Google Gemini...")
        response = llm.tool_call(messages, tools)
        
        print("✅ Google provider working with tool schemas!")
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - {tool_call.function.name}")
        else:
            print(f"Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Google provider: {e}")
        return False

def test_google_provider_simple():
    """Test Google provider without tools"""
    print("\n=== Testing Google Provider (Simple) ===")
    
    try:
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = llm.chat(messages)
        print(f"✅ Simple chat response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error with simple Google chat: {e}")
        return False

def main():
    """Run Google provider tests"""
    print("Testing Google Provider")
    print("=" * 40)
    
    results = {
        "simple_chat": test_google_provider_simple(),
        "tools_chat": test_google_provider_with_tools()
    }
    
    print(f"\n=== Test Results ===")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    main()