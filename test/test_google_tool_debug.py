#!/usr/bin/env python3
"""
Debug test for Google provider tool calling issue
"""

import json
from claude.llm import OAI
from claude.tools.tool_schemas import get_tool_schemas, PREDEFINED_TOOL_SCHEMAS

def test_google_tool_schemas():
    """Test if tool schemas are compatible with Google Gemini API"""
    print("=== Testing Tool Schema Compatibility ===")
    
    # Test with minimal web search schema
    minimal_schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    try:
        # Create Google LLM client
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "search for 'who won ipl 2013'"}
        ]
        
        print("Testing with minimal schema...")
        response = llm.tool_call(messages, [minimal_schema])
        
        print("✅ Minimal schema test successful!")
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                print(f"  Arguments type: {type(tool_call.function.arguments)}")
        else:
            print("No tool calls - response:", response.choices[0].message.content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error with minimal schema: {e}")
        return False

def test_google_no_tools():
    """Test Google provider without tools"""
    print("\n=== Testing Google Provider Without Tools ===")
    
    try:
        # Create Google LLM client
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("Making request without tools...")
        response = llm.run(messages)
        
        print("✅ No-tools test successful!")
        print(f"Response: {response.choices[0].message.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error without tools: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_with_full_schemas():
    """Test Google provider with full tool schemas"""
    print("\n=== Testing Google Provider With Full Schemas ===")
    
    try:
        # Get all tool schemas
        tools = get_tool_schemas()
        
        # Create Google LLM client
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "read the file sum.py"}
        ]
        
        print(f"Testing with {len(tools)} tool schemas...")
        print("First tool schema:", json.dumps(tools[0], indent=2))
        
        response = llm.tool_call(messages, tools)
        
        print("✅ Full schemas test successful!")
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
        else:
            print("No tool calls - response:", response.choices[0].message.content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error with full schemas: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_schema_validation():
    """Debug schema format issues"""
    print("\n=== Debugging Schema Format ===")
    
    # Check if schemas have any issues
    try:
        tools = get_tool_schemas()
        print(f"Generated {len(tools)} tool schemas")
        
        for i, tool in enumerate(tools):
            print(f"\nTool {i+1}: {tool['function']['name']}")
            
            # Check for potential issues
            params = tool['function']['parameters']
            
            # Check for empty required arrays (Google might not like this)
            if 'required' in params and len(params['required']) == 0:
                print(f"  ⚠️  Empty required array")
            
            # Check for missing properties
            if 'properties' in params and len(params['properties']) == 0:
                print(f"  ⚠️  Empty properties")
            
            # Check for array types without items
            for prop_name, prop_def in params.get('properties', {}).items():
                if prop_def.get('type') == 'array' and 'items' not in prop_def:
                    print(f"  ⚠️  Array property '{prop_name}' missing 'items'")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation error: {e}")
        return False

def main():
    """Run all debug tests"""
    print("Google Provider Tool Calling Debug")
    print("=" * 50)
    
    results = {
        "no_tools": test_google_no_tools(),
        "schema_validation": debug_schema_validation(),
        "minimal_schema": test_google_tool_schemas(),
        "full_schemas": test_google_with_full_schemas()
    }
    
    print(f"\n=== Debug Results ===")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    return results

if __name__ == "__main__":
    main()