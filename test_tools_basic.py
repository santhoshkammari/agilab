#!/usr/bin/env python3
"""
Test basic tools functionality from claude.tools
"""

import asyncio
import json
from claude.tools import tools_dict

async def test_read_file():
    """Test reading a file"""
    print("=== Testing Read File ===")
    try:
        result = await tools_dict['read_file'](file_path="/home/ntlpt59/master/own/claude/sum.py")
        print(f"Read file result: {type(result)}")
        if isinstance(result, dict) and 'content' in result:
            print(f"Lines read: {result.get('lines', 'unknown')}")
            print(f"First 100 chars: {result['content'][:100]}...")
        else:
            print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Error in read_file: {e}")
        return False

async def test_list_directory():
    """Test listing directory"""
    print("\n=== Testing List Directory ===")
    try:
        result = await tools_dict['list_directory'](path="/home/ntlpt59/master/own/claude")
        print(f"List directory result: {type(result)}")
        if isinstance(result, list):
            print(f"Items found: {len(result)}")
            print(f"First few items: {result[:5]}")
        else:
            print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Error in list_directory: {e}")
        return False

async def test_write_file():
    """Test writing a file"""
    print("\n=== Testing Write File ===")
    try:
        test_content = "def test_function():\n    return 'Hello from test file!'\n"
        result = await tools_dict['write_file'](
            file_path="/home/ntlpt59/master/own/claude/test_output.py",
            content=test_content
        )
        print(f"Write file result: {result}")
        return True
    except Exception as e:
        print(f"Error in write_file: {e}")
        return False

async def test_bash_execute():
    """Test bash execution"""
    print("\n=== Testing Bash Execute ===")
    try:
        result = await tools_dict['bash_execute'](command="echo 'Hello from bash!'")
        print(f"Bash execute result: {result}")
        return True
    except Exception as e:
        print(f"Error in bash_execute: {e}")
        return False

async def test_glob_find():
    """Test glob find files"""
    print("\n=== Testing Glob Find ===")
    try:
        result = await tools_dict['glob_find_files'](pattern="*.py", path="/home/ntlpt59/master/own/claude")
        print(f"Glob find result: {type(result)}")
        if isinstance(result, list):
            print(f"Python files found: {len(result)}")
            print(f"First few: {result[:3]}")
        else:
            print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Error in glob_find_files: {e}")
        return False

async def test_tools_with_llm():
    """Test tools integration with LLM"""
    print("\n=== Testing Tools with LLM ===")
    try:
        from claude.llm import OAI
        
        # Create LLM instance
        llm = OAI(provider="ollama", base_url="http://localhost:11434/v1", model="qwen3:0.6b")
        
        # Create a simple tool schema for read_file
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }]
        
        messages = [
            {"role": "user", "content": "Please read the file /home/ntlpt59/master/own/claude/sum.py"}
        ]
        
        response = llm.tool_call(messages, tools)
        
        if response.choices[0].message.tool_calls:
            print("Tool call successful!")
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Function called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
            
            # Execute the tool
            args = json.loads(tool_call.function.arguments)
            result = await tools_dict['read_file'](**args)
            print(f"Tool execution result: {type(result)}")
            
        else:
            print("No tool calls made")
            print(f"Response: {response.choices[0].message.content}")
        
        return True
    except Exception as e:
        print(f"Error in tools with LLM: {e}")
        return False

async def main():
    """Run all tool tests"""
    print("Testing Claude Tools")
    print("=" * 40)
    
    test_functions = [
        test_read_file,
        test_list_directory,
        test_write_file,
        test_bash_execute,
        test_glob_find,
        test_tools_with_llm
    ]
    
    results = {}
    for test_func in test_functions:
        try:
            result = await test_func()
            results[test_func.__name__] = result
        except Exception as e:
            print(f"Error in {test_func.__name__}: {e}")
            results[test_func.__name__] = False
    
    print(f"\n=== Test Results ===")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())