import os
import tempfile
from tool_write import write_file, mark_file_as_read, clear_read_state


def test_write_file():
    """Test the write_file function with various scenarios."""
    
    # Clear read state before testing
    clear_read_state()
    
    # Test 1: Basic new file creation
    with tempfile.TemporaryDirectory() as temp_dir:
        new_file = os.path.join(temp_dir, "new_file.txt")
        content = "Hello, World!\nLine 2\nLine 3"
        
        result = write_file(new_file, content)
        assert "Successfully wrote" in result
        assert new_file in result
        
        # Verify file was created with correct content
        with open(new_file, 'r') as f:
            written_content = f.read()
        assert written_content == content
        print("✓ Basic new file creation test passed")
    
    # Test 2: Overwriting existing file after reading
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_file = os.path.join(temp_dir, "existing.txt")
        
        # Create initial file
        with open(existing_file, 'w') as f:
            f.write("Original content")
        
        # Mark as read and then overwrite
        mark_file_as_read(existing_file)
        new_content = "New content after reading"
        
        result = write_file(existing_file, new_content)
        assert "Successfully wrote" in result
        
        # Verify content was overwritten
        with open(existing_file, 'r') as f:
            written_content = f.read()
        assert written_content == new_content
        print("✓ Overwriting after reading test passed")
    
    # Test 3: Attempting to overwrite without reading first
    clear_read_state()
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_file = os.path.join(temp_dir, "existing.txt")
        
        # Create initial file
        with open(existing_file, 'w') as f:
            f.write("Original content")
        
        # Try to overwrite without reading first
        try:
            write_file(existing_file, "Should fail")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "File must be read before writing" in str(e)
            print("✓ Read-before-write policy test passed")
    
    # Test 4: Relative path validation
    try:
        write_file("relative/path.txt", "content")
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e)
        print("✓ Relative path validation test passed")
    
    # Test 5: Parent directory doesn't exist
    try:
        write_file("/nonexistent/directory/file.txt", "content")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Parent directory does not exist" in str(e)
        print("✓ Parent directory validation test passed")
    
    # Test 6: Empty content
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_file = os.path.join(temp_dir, "empty.txt")
        
        result = write_file(empty_file, "")
        assert "Successfully wrote 0 bytes" in result
        
        # Verify file exists and is empty
        assert os.path.exists(empty_file)
        with open(empty_file, 'r') as f:
            content = f.read()
        assert content == ""
        print("✓ Empty content test passed")
    
    # Test 7: Large content
    with tempfile.TemporaryDirectory() as temp_dir:
        large_file = os.path.join(temp_dir, "large.txt")
        large_content = "A" * 10000 + "\n" + "B" * 5000
        
        result = write_file(large_file, large_content)
        assert "Successfully wrote" in result
        
        # Verify large content was written correctly
        with open(large_file, 'r') as f:
            written_content = f.read()
        assert written_content == large_content
        print("✓ Large content test passed")
    
    # Test 8: Special characters and unicode
    with tempfile.TemporaryDirectory() as temp_dir:
        unicode_file = os.path.join(temp_dir, "unicode.txt")
        unicode_content = "Hello 世界! 🌍\nSpecial chars: àáâãäå\nSymbols: €£¥"
        
        result = write_file(unicode_file, unicode_content)
        assert "Successfully wrote" in result
        
        # Verify unicode content was written correctly
        with open(unicode_file, 'r', encoding='utf-8') as f:
            written_content = f.read()
        assert written_content == unicode_content
        print("✓ Unicode content test passed")
    
    # Test 9: Code content with proper formatting
    with tempfile.TemporaryDirectory() as temp_dir:
        code_file = os.path.join(temp_dir, "code.py")
        code_content = '''def hello_world():
    """Print hello world message."""
    print("Hello, World!")
    
    if True:
        return "success"

class MyClass:
    def __init__(self):
        self.value = 42'''
        
        result = write_file(code_file, code_content)
        assert "Successfully wrote" in result
        
        # Verify code formatting was preserved
        with open(code_file, 'r') as f:
            written_content = f.read()
        assert written_content == code_content
        print("✓ Code content formatting test passed")
    
    # Test 10: JSON content
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file = os.path.join(temp_dir, "config.json")
        json_content = '''{
  "name": "MyApp",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.0",
    "react": "^18.0.0"
  },
  "scripts": {
    "start": "node server.js",
    "test": "jest"
  }
}'''
        
        result = write_file(json_file, json_content)
        assert "Successfully wrote" in result
        
        # Verify JSON content was written correctly
        with open(json_file, 'r') as f:
            written_content = f.read()
        assert written_content == json_content
        print("✓ JSON content test passed")
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_write_file()