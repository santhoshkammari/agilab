import os
import tempfile
import shutil
from tool_ls import list_directory


def test_list_directory():
    """Test the list_directory function with various scenarios."""
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Basic directory listing
        # Create some test files and directories
        test_file1 = os.path.join(temp_dir, "file1.txt")
        test_file2 = os.path.join(temp_dir, "file2.log")
        test_dir1 = os.path.join(temp_dir, "subdir1")
        test_dir2 = os.path.join(temp_dir, "subdir2")
        
        with open(test_file1, 'w') as f:
            f.write("test content")
        with open(test_file2, 'w') as f:
            f.write("log content")
        os.mkdir(test_dir1)
        os.mkdir(test_dir2)
        
        result = list_directory(temp_dir)
        assert "entries" in result, "Result should contain 'entries' key"
        assert len(result["entries"]) == 4, f"Should have 4 entries, got {len(result['entries'])}"
        
        # Check that directories come first
        entries = result["entries"]
        assert entries[0]["type"] == "directory", "First entry should be a directory"
        assert entries[1]["type"] == "directory", "Second entry should be a directory"
        assert entries[2]["type"] == "file", "Third entry should be a file"
        assert entries[3]["type"] == "file", "Fourth entry should be a file"
        
        print("✓ Basic directory listing test passed")
        
        # Test 2: Directory listing with ignore patterns
        result = list_directory(temp_dir, ignore=["*.log", "subdir1"])
        entries = result["entries"]
        entry_names = [entry["name"] for entry in entries]
        
        assert "file2.log" not in entry_names, "*.log files should be ignored"
        assert "subdir1" not in entry_names, "subdir1 should be ignored"
        assert "file1.txt" in entry_names, "file1.txt should not be ignored"
        assert "subdir2" in entry_names, "subdir2 should not be ignored"
        
        print("✓ Ignore patterns test passed")
        
        # Test 3: Empty directory
        empty_dir = os.path.join(temp_dir, "empty")
        os.mkdir(empty_dir)
        
        result = list_directory(empty_dir)
        assert result["entries"] == [], "Empty directory should return empty entries list"
        
        print("✓ Empty directory test passed")
        
        # Test 4: Entry properties validation
        result = list_directory(temp_dir)
        file_entry = None
        dir_entry = None
        
        for entry in result["entries"]:
            if entry["type"] == "file" and entry["name"] == "file1.txt":
                file_entry = entry
            elif entry["type"] == "directory" and entry["name"] == "subdir2":
                dir_entry = entry
        
        assert file_entry is not None, "Should find file1.txt entry"
        assert dir_entry is not None, "Should find subdir2 entry"
        
        # Check file entry properties
        assert "name" in file_entry, "File entry should have 'name'"
        assert "type" in file_entry, "File entry should have 'type'"
        assert "size" in file_entry, "File entry should have 'size'"
        assert "modified" in file_entry, "File entry should have 'modified'"
        assert file_entry["type"] == "file", "File entry type should be 'file'"
        assert isinstance(file_entry["size"], int), "File size should be integer"
        assert file_entry["modified"].endswith("Z"), "Modified time should end with 'Z'"
        
        # Check directory entry properties
        assert dir_entry["type"] == "directory", "Directory entry type should be 'directory'"
        assert dir_entry["size"] is None, "Directory size should be None"
        
        print("✓ Entry properties validation test passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    # Test 5: File not found
    try:
        list_directory("/nonexistent/directory")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found test passed")
    
    # Test 6: Relative path validation
    try:
        list_directory("relative/path")
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e), f"Error message should mention absolute path: {e}"
        print("✓ Relative path validation test passed")
    
    # Test 7: Not a directory error
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        list_directory(temp_file.name)
        assert False, "Should have raised NotADirectoryError"
    except NotADirectoryError:
        print("✓ Not a directory test passed")
    finally:
        os.unlink(temp_file.name)
    
    # Test 8: Multiple ignore patterns
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test files with different extensions
        files = ["test.txt", "debug.log", "cache.tmp", "data.json", "backup.bak"]
        for filename in files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("content")
        
        result = list_directory(temp_dir, ignore=["*.log", "*.tmp", "*.bak"])
        entry_names = [entry["name"] for entry in result["entries"]]
        
        assert "test.txt" in entry_names, "test.txt should not be ignored"
        assert "data.json" in entry_names, "data.json should not be ignored"
        assert "debug.log" not in entry_names, "debug.log should be ignored"
        assert "cache.tmp" not in entry_names, "cache.tmp should be ignored"
        assert "backup.bak" not in entry_names, "backup.bak should be ignored"
        
        print("✓ Multiple ignore patterns test passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_list_directory()