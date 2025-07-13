#!/usr/bin/env python3
"""
Simple script to run the EditTool function and print raw results
"""

import asyncio
import tempfile
import json
from claude.tools.__files_advanced import EditTool


async def run_edit_tool():
    """Run the edit tool and print raw results"""
    
    # Create EditTool instance
    edit_tool = EditTool()
    
    # Create a temporary file with sample content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_content = """Hello World!
This is line 2.
Hello again on line 3.
Final line here."""
        f.write(test_content)
        temp_file = f.name
    
    print("Original file content:")
    print("-" * 40)
    with open(temp_file, 'r') as f:
        print(f.read())
    print("-" * 40)
    
    print("\nRunning edit_file function...")
    print("Parameters:")
    print(f"  file_path: {temp_file}")
    print(f"  old_string: 'Hello World!'")
    print(f"  new_string: 'Hi Universe!'")
    print(f"  replace_all: False")
    
    # Run the edit function
    result = await edit_tool.edit_file(
        file_path=temp_file,
        old_string="Hello World!",
        new_string="Hi Universe!",
        replace_all=False
    )
    
    print("\nRaw result from edit_file():")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print("=" * 50)
    
    print("\nFile content after edit:")
    print("-" * 40)
    with open(temp_file, 'r') as f:
        print(f.read())
    print("-" * 40)
    
    # Clean up
    import os
    os.unlink(temp_file)


if __name__ == "__main__":
    asyncio.run(run_edit_tool())