import os
import tempfile
from tool_multiedit import multi_edit, create_file_with_multi_edit


def test_multi_edit():
    """Test the multi_edit function with various scenarios."""
    
    # Test 1: Basic single edit
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world\nThis is a test\n")
        temp_file = f.name
    
    try:
        edits = [{"old_string": "Hello world", "new_string": "Hello universe"}]
        result = multi_edit(temp_file, edits)
        assert "Successfully applied 1 edits" in result
        
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "Hello universe" in content
        assert "Hello world" not in content
        print("✓ Basic single edit test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 2: Multiple sequential edits
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.js') as f:
        f.write("const oldName = 'value';\nconsole.log(oldName);\n")
        temp_file = f.name
    
    try:
        edits = [
            {"old_string": "const oldName = 'value';", "new_string": "const newName = 'value';"},
            {"old_string": "console.log(oldName);", "new_string": "console.log(newName);"}
        ]
        result = multi_edit(temp_file, edits)
        assert "Successfully applied 2 edits" in result
        
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "newName" in content
        assert "oldName" not in content
        print("✓ Multiple sequential edits test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 3: Replace all occurrences
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("def old_func():\n    pass\n\ndef test_old_func():\n    old_func()\n")
        temp_file = f.name
    
    try:
        edits = [{"old_string": "old_func", "new_string": "new_func", "replace_all": True}]
        result = multi_edit(temp_file, edits)
        assert "Successfully applied 1 edits" in result
        
        with open(temp_file, 'r') as f:
            content = f.read()
        assert content.count("new_func") == 3
        assert "old_func" not in content
        print("✓ Replace all occurrences test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 4: Empty edits list
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        temp_file = f.name
    
    try:
        try:
            multi_edit(temp_file, [])
            assert False, "Should have raised ValueError for empty edits"
        except ValueError as e:
            assert "edits list cannot be empty" in str(e)
            print("✓ Empty edits list test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 5: Relative path validation
    try:
        multi_edit("relative/path.txt", [{"old_string": "a", "new_string": "b"}])
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e)
        print("✓ Relative path validation test passed")
    
    # Test 6: File not found
    try:
        multi_edit("/nonexistent/file.txt", [{"old_string": "a", "new_string": "b"}])
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found test passed")
    
    # Test 7: String not found
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world")
        temp_file = f.name
    
    try:
        try:
            edits = [{"old_string": "nonexistent", "new_string": "replacement"}]
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError for string not found"
        except ValueError as e:
            assert "String not found" in str(e)
            print("✓ String not found test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 8: Multiple matches without replace_all
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test test test")
        temp_file = f.name
    
    try:
        try:
            edits = [{"old_string": "test", "new_string": "replacement"}]
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError for multiple matches"
        except ValueError as e:
            assert "appears 3 times" in str(e)
            print("✓ Multiple matches without replace_all test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 9: Same old_string and new_string
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world")
        temp_file = f.name
    
    try:
        try:
            edits = [{"old_string": "Hello", "new_string": "Hello"}]
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError for same strings"
        except ValueError as e:
            assert "must be different" in str(e)
            print("✓ Same old_string and new_string test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 10: Rollback on failure
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        original_content = "Line 1\nLine 2\nLine 3\n"
        f.write(original_content)
        temp_file = f.name
    
    try:
        try:
            edits = [
                {"old_string": "Line 1", "new_string": "Modified Line 1"},
                {"old_string": "nonexistent", "new_string": "replacement"}  # This will fail
            ]
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError"
        except ValueError:
            # Check that file was rolled back to original content
            with open(temp_file, 'r') as f:
                content = f.read()
            assert content == original_content, "File should be rolled back to original content"
            print("✓ Rollback on failure test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 11: Invalid edit structure
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        temp_file = f.name
    
    try:
        try:
            edits = ["invalid_edit"]  # Not a dictionary
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError for invalid edit structure"
        except ValueError as e:
            assert "must be a dictionary" in str(e)
            print("✓ Invalid edit structure test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 12: Missing required fields
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        temp_file = f.name
    
    try:
        try:
            edits = [{"old_string": "test"}]  # Missing new_string
            multi_edit(temp_file, edits)
            assert False, "Should have raised ValueError for missing fields"
        except ValueError as e:
            assert "must contain 'old_string' and 'new_string'" in str(e)
            print("✓ Missing required fields test passed")
    finally:
        os.unlink(temp_file)


def test_create_file_with_multi_edit():
    """Test the create_file_with_multi_edit function."""
    
    # Test 1: Create new file
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "new_file.txt")
    
    try:
        content = "This is a new file\nWith multiple lines\n"
        result = create_file_with_multi_edit(temp_file, content)
        assert "Successfully applied 1 edits" in result
        
        with open(temp_file, 'r') as f:
            file_content = f.read()
        assert file_content == content
        print("✓ Create new file test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        os.rmdir(temp_dir)
    
    # Test 2: Create file with directory creation
    temp_dir = tempfile.mkdtemp()
    new_dir = os.path.join(temp_dir, "subdir", "nested")
    temp_file = os.path.join(new_dir, "new_file.txt")
    
    try:
        content = "File in nested directory"
        result = create_file_with_multi_edit(temp_file, content)
        assert "Successfully applied 1 edits" in result
        assert os.path.exists(temp_file)
        print("✓ Create file with directory creation test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        import shutil
        shutil.rmtree(temp_dir)
    
    # Test 3: File already exists
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = f.name
    
    try:
        try:
            create_file_with_multi_edit(temp_file, "content")
            assert False, "Should have raised ValueError for existing file"
        except ValueError as e:
            assert "File already exists" in str(e)
            print("✓ File already exists test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 4: Relative path validation
    try:
        create_file_with_multi_edit("relative/path.txt", "content")
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e)
        print("✓ Relative path validation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running MultiEdit tool tests...\n")
    
    test_multi_edit()
    print()
    test_create_file_with_multi_edit()
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()