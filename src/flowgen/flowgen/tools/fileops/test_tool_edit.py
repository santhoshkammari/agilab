import os
import tempfile
from tool_edit import edit_file, mark_file_as_read, clear_read_files


def test_edit_file():
    """Test the edit_file function with various scenarios."""
    
    # Clear any previous read file tracking
    clear_read_files()
    
    # Test 1: Basic string replacement
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world\nThis is a test\nHello again")
        temp_file = f.name
    
    try:
        # Mark file as read first
        mark_file_as_read(temp_file)
        
        result = edit_file(temp_file, "Hello world", "Hi world")
        assert "Replaced text" in result
        
        # Verify the change
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "Hi world" in content
        assert "Hello world" not in content
        assert "Hello again" in content  # Other occurrences should remain
        print("✓ Basic string replacement test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 2: Multi-line string replacement
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("function old() {\n    return 'old';\n}\nother code")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        old_func = "function old() {\n    return 'old';\n}"
        new_func = "function new() {\n    return 'new';\n}"
        
        result = edit_file(temp_file, old_func, new_func)
        assert "Replaced text" in result
        
        # Verify the change
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "function new()" in content
        assert "return 'new'" in content
        assert "other code" in content
        print("✓ Multi-line string replacement test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 3: Replace all occurrences
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("var oldName = 1;\nvar oldName = 2;\nvar oldName = 3;")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        result = edit_file(temp_file, "oldName", "newName", replace_all=True)
        assert "Replaced all 3 occurrences" in result
        
        # Verify all changes
        with open(temp_file, 'r') as f:
            content = f.read()
        assert content.count("newName") == 3
        assert "oldName" not in content
        print("✓ Replace all occurrences test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 4: Content deletion (empty new_string)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Keep this\nDELETE THIS LINE\nKeep this too")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        result = edit_file(temp_file, "DELETE THIS LINE\n", "")
        assert "Replaced text" in result
        
        # Verify deletion
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "DELETE THIS LINE" not in content
        assert "Keep this\nKeep this too" == content
        print("✓ Content deletion test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 5: File not read first (should fail)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        temp_file = f.name
    
    try:
        # Don't mark as read
        clear_read_files()
        
        try:
            edit_file(temp_file, "test", "new")
            assert False, "Should have raised RuntimeError for file not read first"
        except RuntimeError as e:
            assert "must be read before editing" in str(e)
            print("✓ File not read first test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 6: String not found (should fail)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        try:
            edit_file(temp_file, "nonexistent", "replacement")
            assert False, "Should have raised RuntimeError for string not found"
        except RuntimeError as e:
            assert "String not found" in str(e)
            print("✓ String not found test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 7: Multiple matches without replace_all (should fail)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test\ntest\ntest")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        try:
            edit_file(temp_file, "test", "replacement")
            assert False, "Should have raised RuntimeError for multiple matches"
        except RuntimeError as e:
            assert "Multiple matches found" in str(e)
            assert "replace_all=True" in str(e)
            print("✓ Multiple matches without replace_all test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 8: Identical strings (should fail)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello world")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        try:
            edit_file(temp_file, "Hello", "Hello")
            assert False, "Should have raised ValueError for identical strings"
        except ValueError as e:
            assert "identical" in str(e)
            print("✓ Identical strings test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 9: Relative path validation (should fail)
    try:
        edit_file("relative/path.txt", "old", "new")
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e)
        print("✓ Relative path validation test passed")
    
    # Test 10: File not found (should fail)
    try:
        mark_file_as_read("/nonexistent/file.txt")
        edit_file("/nonexistent/file.txt", "old", "new")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found test passed")
    
    # Test 11: Whitespace preservation
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("def func():\n    old_variable = 1\n    return old_variable")
        temp_file = f.name
    
    try:
        mark_file_as_read(temp_file)
        
        result = edit_file(temp_file, "    old_variable = 1", "    new_variable = 1")
        assert "Replaced text" in result
        
        # Verify indentation is preserved
        with open(temp_file, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        assert lines[1] == "    new_variable = 1"
        assert "    return old_variable" in content  # Other instance remains
        print("✓ Whitespace preservation test passed")
    finally:
        os.unlink(temp_file)
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_edit_file()