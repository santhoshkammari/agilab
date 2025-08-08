#!/usr/bin/env python3
"""
Test script for the EditTool functionality from claude/tools/__files_advanced.py

This script tests various edit operations including:
- Basic string replacement
- Replace all occurrences
- Error handling for missing strings
- Multiple occurrence detection
"""

import asyncio
import os
import tempfile
from pathlib import Path

from claude.tools.files import EditTool, ValidationError, FileSystemError


async def test_basic_edit():
    """Test basic string replacement"""
    print("Testing basic string replacement...")
    
    edit_tool = EditTool()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = """Hello world!
This is a test file.
Hello again!"""
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Test single replacement
        result = await edit_tool.edit_file(
            file_path=temp_file,
            old_string="Hello world!",
            new_string="Hi there!"
        )
        
        print(f"✓ Basic edit successful: {result['replacements']} replacement(s)")
        print(f"  File: {result['file_path']}")
        print(f"  Size change: {result['original_size']} → {result['new_size']}")
        
        # Verify the change
        with open(temp_file, 'r') as f:
            new_content = f.read()
        
        assert "Hi there!" in new_content
        assert "Hello world!" not in new_content
        print("✓ Content verification passed")
        
    finally:
        os.unlink(temp_file)


async def test_replace_all():
    """Test replacing all occurrences"""
    print("\nTesting replace all occurrences...")
    
    edit_tool = EditTool()
    
    # Create a temporary file with multiple occurrences
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = """foo bar foo
Another foo here
And one more foo"""
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Test replace all
        result = await edit_tool.edit_file(
            file_path=temp_file,
            old_string="foo",
            new_string="baz",
            replace_all=True
        )
        
        print(f"✓ Replace all successful: {result['replacements']} replacement(s)")
        
        # Verify all occurrences were replaced
        with open(temp_file, 'r') as f:
            new_content = f.read()
        
        assert "foo" not in new_content
        assert new_content.count("baz") == 4
        print("✓ All occurrences replaced correctly")
        
    finally:
        os.unlink(temp_file)


async def test_multiple_occurrence_error():
    """Test error handling for multiple occurrences without replace_all"""
    print("\nTesting multiple occurrence error handling...")
    
    edit_tool = EditTool()
    
    # Create a temporary file with multiple occurrences
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = """test line
another test line
final test"""
        f.write(test_content)
        temp_file = f.name
    
    try:
        # This should raise an error due to multiple occurrences
        try:
            await edit_tool.edit_file(
                file_path=temp_file,
                old_string="test",
                new_string="example"
            )
            assert False, "Should have raised FileSystemError"
        except FileSystemError as e:
            print(f"✓ Correctly caught multiple occurrence error: {str(e)}")
            assert "Multiple occurrences" in str(e)
        
    finally:
        os.unlink(temp_file)


async def test_string_not_found():
    """Test error handling for string not found"""
    print("\nTesting string not found error...")
    
    edit_tool = EditTool()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = "This is a simple test file."
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Try to replace a string that doesn't exist
        try:
            await edit_tool.edit_file(
                file_path=temp_file,
                old_string="nonexistent",
                new_string="replacement"
            )
            assert False, "Should have raised FileSystemError"
        except FileSystemError as e:
            print(f"✓ Correctly caught string not found error: {str(e)}")
            assert "String not found" in str(e)
        
    finally:
        os.unlink(temp_file)


async def test_file_not_found():
    """Test error handling for non-existent file"""
    print("\nTesting file not found error...")
    
    edit_tool = EditTool()
    
    try:
        await edit_tool.edit_file(
            file_path="/nonexistent/file.txt",
            old_string="test",
            new_string="replacement"
        )
        assert False, "Should have raised FileSystemError"
    except FileSystemError as e:
        print(f"✓ Correctly caught file not found error: {str(e)}")
        assert "File does not exist" in str(e)


async def test_code_file_edit():
    """Test editing a Python code file"""
    print("\nTesting Python code file editing...")
    
    edit_tool = EditTool()
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        test_code = '''def hello():
    print("Hello World")
    return "success"

def goodbye():
    print("Goodbye World")
    return "done"
'''
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Replace function content
        result = await edit_tool.edit_file(
            file_path=temp_file,
            old_string='    print("Hello World")',
            new_string='    print("Hi there!")'
        )
        
        print(f"✓ Code edit successful: {result['replacements']} replacement(s)")
        
        # Verify the change
        with open(temp_file, 'r') as f:
            new_content = f.read()
        
        assert 'print("Hi there!")' in new_content
        assert 'print("Hello World")' not in new_content
        print("✓ Code content verification passed")
        
    finally:
        os.unlink(temp_file)


async def test_whitespace_sensitive():
    """Test that edits are whitespace-sensitive"""
    print("\nTesting whitespace sensitivity...")
    
    edit_tool = EditTool()
    
    # Create a file with specific whitespace
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = """line1
    indented line
        double indented
line4"""
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Replace the exact indented line
        result = await edit_tool.edit_file(
            file_path=temp_file,
            old_string="    indented line",
            new_string="    modified line"
        )
        
        print(f"✓ Whitespace-sensitive edit successful")
        
        # Verify exact whitespace preservation
        with open(temp_file, 'r') as f:
            new_content = f.read()
        
        assert "    modified line" in new_content
        assert "    indented line" not in new_content
        print("✓ Whitespace preservation verified")
        
    finally:
        os.unlink(temp_file)


async def run_all_tests():
    """Run all edit tool tests"""
    print("Starting EditTool tests...\n")
    
    tests = [
        test_basic_edit,
        test_replace_all,
        test_multiple_occurrence_error,
        test_string_not_found,
        test_file_not_found,
        test_code_file_edit,
        test_whitespace_sensitive
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)