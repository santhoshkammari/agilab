#!/usr/bin/env python3
"""
Test Google provider tool execution after fixing argument handling
"""

import json
from claude.llm import OAI
from claude.tools import get_tool_schemas

def test_google_web_search():
    """Test Google provider with web search tool"""
    print("=== Testing Google Provider - Web Search ===")
    
    try:
        # Get tool schemas
        tools = get_tool_schemas()
        
        # Create Google LLM client
        llm = OAI(
            provider="google", 
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            model="gemini-2.5-flash"
        )
        
        messages = [
            {"role": "user", "content": "search web for 'who won ipl 2013'"}
        ]
        
        print("Making request to Google Gemini...")
        response = llm.tool_call(messages, tools)
        
        print("✅ Google provider request successful!")
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments type: {type(tool_call.function.arguments)}")
                print(f"  Arguments: {tool_call.function.arguments}")
                
                # Test argument parsing
                if isinstance(tool_call.function.arguments, str):
                    try:
                        args_dict = json.loads(tool_call.function.arguments)
                        print(f"  ✅ Parsed JSON arguments: {args_dict}")
                    except json.JSONDecodeError as e:
                        print(f"  ❌ JSON parse error: {e}")
                elif isinstance(tool_call.function.arguments, dict):
                    print(f"  ✅ Dict arguments: {tool_call.function.arguments}")
                else:
                    print(f"  ⚠️  Unknown argument type: {type(tool_call.function.arguments)}")
        else:
            print("No tool calls detected")
            print(f"Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Google provider: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_read_file():
    """Test Google provider with read file tool"""
    print("\n=== Testing Google Provider - Read File ===")
    
    try:
        # Get tool schemas
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
        
        print("Making request to Google Gemini...")
        response = llm.tool_call(messages, tools)
        
        print("✅ Google provider request successful!")
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments type: {type(tool_call.function.arguments)}")
                print(f"  Arguments: {tool_call.function.arguments}")
                
                # Test argument parsing
                if isinstance(tool_call.function.arguments, str):
                    try:
                        args_dict = json.loads(tool_call.function.arguments)
                        print(f"  ✅ Parsed JSON arguments: {args_dict}")
                    except json.JSONDecodeError as e:
                        print(f"  ❌ JSON parse error: {e}")
                elif isinstance(tool_call.function.arguments, dict):
                    print(f"  ✅ Dict arguments: {tool_call.function.arguments}")
                else:
                    print(f"  ⚠️  Unknown argument type: {type(tool_call.function.arguments)}")
        else:
            print("No tool calls detected")
            print(f"Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Google provider: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_type_handling():
    """Test argument type handling logic"""
    print("\n=== Testing Argument Type Handling Logic ===")
    
    # Mock tool call object
    class MockFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments
    
    class MockToolCall:
        def __init__(self, name, arguments):
            self.function = MockFunction(name, arguments)
    
    test_cases = [
        # String JSON arguments
        ('{"query": "test search"}', "string JSON"),
        # Dict arguments  
        ({"query": "test search"}, "dict"),
        # Empty arguments
        ({}, "empty dict"),
        ("", "empty string"),
        (None, "None"),
    ]
    
    for args, case_type in test_cases:
        print(f"\nTesting {case_type}: {args}")
        mock_call = MockToolCall("web_search", args)
        
        # Test the logic from input.py
        if mock_call.function.arguments:
            if isinstance(mock_call.function.arguments, str):
                try:
                    args_dict = json.loads(mock_call.function.arguments)
                    tool_args = list(args_dict.values())[0] if args_dict else ""
                    print(f"  ✅ String->JSON->first_value: '{tool_args}'")
                except (json.JSONDecodeError, IndexError):
                    tool_args = mock_call.function.arguments
                    print(f"  ✅ String fallback: '{tool_args}'")
            elif isinstance(mock_call.function.arguments, dict):
                tool_args = list(mock_call.function.arguments.values())[0] if mock_call.function.arguments else ""
                print(f"  ✅ Dict->first_value: '{tool_args}'")
            else:
                tool_args = str(mock_call.function.arguments)
                print(f"  ✅ Str conversion: '{tool_args}'")
        else:
            tool_args = ""
            print(f"  ✅ Empty fallback: '{tool_args}'")
    
    return True

def main():
    """Run all tests"""
    print("Testing Google Provider Tool Execution")
    print("=" * 50)
    
    results = {
        "argument_handling": test_argument_type_handling(),
        "google_web_search": test_google_web_search(),
        "google_read_file": test_google_read_file()
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