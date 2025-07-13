#!/usr/bin/env python3
"""
Test the EditTool with diff output - formatted for better readability
"""

import asyncio
import tempfile
import json
from claude.tools.__files_advanced import EditTool


async def test_edit_with_diff():
    """Test edit tool and display the diff clearly"""
    
    edit_tool = EditTool()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        test_content = '''def hello():
    print("Hello World")
    return "success"

def goodbye():
    print("Goodbye World")
    return "done"
'''
        f.write(test_content)
        temp_file = f.name
    
    print("Original file content:")
    print("=" * 50)
    with open(temp_file, 'r') as f:
        content = f.read()
        for i, line in enumerate(content.split('\n'), 1):
            print(f"{i:2}: {line}")
    print("=" * 50)
    
    # Perform edit
    result = await edit_tool.edit_file(
        file_path=temp_file,
        old_string='    print("Hello World")',
        new_string='    print("Hi Universe!")',
        replace_all=False
    )
    
    print("\nEdit operation result:")
    print("-" * 30)
    print(f"Success: {result['success']}")
    print(f"Replacements: {result['replacements']}")
    print(f"Old string: {result['old_string']}")
    print(f"New string: {result['new_string']}")
    print(f"Size change: {result['original_size']} → {result['new_size']}")
    
    print("\nDiff output:")
    print("=" * 50)
    print(result['diff'])
    print("=" * 50)
    
    print("\nFile content after edit:")
    print("=" * 50)
    with open(temp_file, 'r') as f:
        content = f.read()
        for i, line in enumerate(content.split('\n'), 1):
            print(f"{i:2}: {line}")
    print("=" * 50)
    
    # Clean up
    import os
    os.unlink(temp_file)


if __name__ == "__main__":
    asyncio.run(test_edit_with_diff())