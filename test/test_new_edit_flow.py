#!/usr/bin/env python3
"""
Test the new edit flow with diff confirmation
"""

import asyncio
import tempfile
import json
from claude.tools.files import EditTool


async def test_new_edit_flow():
    """Test the new edit flow that shows diff before applying"""
    
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
    
    # Step 1: Execute edit (should not write file yet)
    print("\n1. Executing edit_file (should prepare edit but not apply)...")
    result = await edit_tool.edit_file(
        file_path=temp_file,
        old_string='    print("Hello World")',
        new_string='    print("Hi Universe!")',
        replace_all=False
    )
    
    print("Edit result:")
    print(json.dumps(result, indent=2))
    
    # Check if file was modified (it shouldn't be)
    print("\n2. Checking if file was modified (should be unchanged)...")
    with open(temp_file, 'r') as f:
        current_content = f.read()
    
    original_content = '''def hello():
    print("Hello World")
    return "success"

def goodbye():
    print("Goodbye World")
    return "done"
'''
    
    if current_content == original_content:
        print("✓ File unchanged (correct - edit is pending)")
    else:
        print("✗ File was modified (incorrect - should be pending)")
    
    print("\nDiff from result:")
    print("-" * 30)
    print(result.get('diff', 'No diff'))
    print("-" * 30)
    
    # Step 3: Apply the edit
    print("\n3. Applying the pending edit...")
    apply_result = await edit_tool.apply_pending_edit()
    print("Apply result:")
    print(json.dumps(apply_result, indent=2))
    
    # Check if file was modified now
    print("\n4. Checking if file was modified after apply...")
    with open(temp_file, 'r') as f:
        final_content = f.read()
        for i, line in enumerate(final_content.split('\n'), 1):
            print(f"{i:2}: {line}")
    
    if 'Hi Universe!' in final_content and 'Hello World' not in final_content:
        print("✓ File correctly modified after apply")
    else:
        print("✗ File was not modified correctly")
    
    # Test discard functionality
    print("\n5. Testing discard functionality...")
    
    # Make another edit
    result2 = await edit_tool.edit_file(
        file_path=temp_file,
        old_string='    print("Hi Universe!")',
        new_string='    print("Bonjour!")',
        replace_all=False
    )
    
    # Discard it
    discard_result = await edit_tool.discard_pending_edit()
    print("Discard result:")
    print(json.dumps(discard_result, indent=2))
    
    # Check file is unchanged
    with open(temp_file, 'r') as f:
        unchanged_content = f.read()
    
    if unchanged_content == final_content:
        print("✓ File unchanged after discard (correct)")
    else:
        print("✗ File was modified after discard (incorrect)")
    
    # Clean up
    import os
    os.unlink(temp_file)
    
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    asyncio.run(test_new_edit_flow())