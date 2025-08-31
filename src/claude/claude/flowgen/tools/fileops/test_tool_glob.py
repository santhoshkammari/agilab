import os
import tempfile
import time
from tool_glob import glob_files, glob_files_advanced


def test_glob_files():
    """Test the glob_files function with various scenarios."""
    
    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test 1: Basic glob pattern
        # Create some test files
        test_files = [
            "test1.py",
            "test2.py", 
            "script.js",
            "README.md",
            "config.json"
        ]
        
        created_files = []
        for filename in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Content of {filename}")
            created_files.append(filepath)
            time.sleep(0.01)  # Small delay to ensure different modification times
        
        # Test Python files pattern
        result = glob_files("*.py", temp_dir)
        py_files = [f for f in created_files if f.endswith('.py')]
        assert len(result) == 2, f"Expected 2 Python files, got {len(result)}"
        assert all(f.endswith('.py') for f in result), "All results should be Python files"
        print("✓ Basic Python files glob test passed")
        
        # Test all files pattern
        result = glob_files("*", temp_dir)
        assert len(result) == len(test_files), f"Expected {len(test_files)} files, got {len(result)}"
        print("✓ All files glob test passed")
        
        # Test 2: Recursive pattern with subdirectories
        # Create subdirectory structure
        subdir = os.path.join(temp_dir, "src")
        os.makedirs(subdir)
        
        subdir_files = [
            "main.py",
            "utils.py",
            "data.json"
        ]
        
        for filename in subdir_files:
            filepath = os.path.join(subdir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Content of {filename}")
            time.sleep(0.01)
        
        # Test recursive Python files
        result = glob_files("**/*.py", temp_dir)
        total_py_files = 4  # 2 in root + 2 in subdir
        assert len(result) >= total_py_files, f"Expected at least {total_py_files} Python files, got {len(result)}"
        print("✓ Recursive Python files glob test passed")
        
        # Test 3: Multiple extensions pattern
        result = glob_files("*.{py,js,json}", temp_dir)
        expected_extensions = {'.py', '.js', '.json'}
        result_extensions = {os.path.splitext(f)[1] for f in result}
        assert result_extensions.issubset(expected_extensions), f"Unexpected extensions: {result_extensions}"
        print("✓ Multiple extensions glob test passed")
        
        # Test 4: No matches
        result = glob_files("*.xyz", temp_dir)
        assert len(result) == 0, f"Expected no matches, got {len(result)}"
        print("✓ No matches glob test passed")
        
        # Test 5: Current directory (no path specified)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = glob_files("*.py")
            assert len(result) == 2, f"Expected 2 Python files in current dir, got {len(result)}"
            print("✓ Current directory glob test passed")
        finally:
            os.chdir(original_cwd)
    
    # Test 6: Error cases
    
    # Empty pattern
    try:
        glob_files("")
        assert False, "Should have raised ValueError for empty pattern"
    except ValueError as e:
        assert "Pattern cannot be empty" in str(e)
        print("✓ Empty pattern test passed")
    
    # Invalid path
    try:
        glob_files("*.py", "/nonexistent/directory")
        assert False, "Should have raised ValueError for invalid path"
    except ValueError as e:
        assert "Path does not exist" in str(e)
        print("✓ Invalid path test passed")
    
    # Path is not a directory (create a file and try to use it as directory)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
    
    try:
        try:
            glob_files("*.py", temp_file)
            assert False, "Should have raised ValueError for file path"
        except ValueError as e:
            assert "Path is not a directory" in str(e)
            print("✓ File path test passed")
    finally:
        os.unlink(temp_file)


def test_glob_files_advanced():
    """Test the advanced glob_files function with additional options."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create test structure
        files = ["file1.py", "file2.js", "README.md"]
        dirs = ["src", "tests", "docs"]
        
        # Create files
        for filename in files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Content of {filename}")
            time.sleep(0.01)
        
        # Create directories
        for dirname in dirs:
            os.makedirs(os.path.join(temp_dir, dirname))
        
        # Test 1: Include directories
        result_files_only = glob_files_advanced("*", temp_dir, include_dirs=False)
        result_with_dirs = glob_files_advanced("*", temp_dir, include_dirs=True)
        
        assert len(result_files_only) == len(files), f"Expected {len(files)} files only"
        assert len(result_with_dirs) == len(files) + len(dirs), f"Expected {len(files) + len(dirs)} total items"
        print("✓ Include directories test passed")
        
        # Test 2: Max results limit
        result_limited = glob_files_advanced("*", temp_dir, include_dirs=True, max_results=2)
        assert len(result_limited) == 2, f"Expected 2 results with limit, got {len(result_limited)}"
        print("✓ Max results limit test passed")
        
        # Test 3: Max results with no limit
        result_unlimited = glob_files_advanced("*", temp_dir, include_dirs=True, max_results=None)
        assert len(result_unlimited) == len(files) + len(dirs), "Unlimited should return all results"
        print("✓ Unlimited results test passed")


def test_sorting_behavior():
    """Test that files are sorted by modification time (newest first)."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create files with deliberate time gaps
        files = ["old_file.py", "newer_file.py", "newest_file.py"]
        created_paths = []
        
        for filename in files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Content of {filename}")
            created_paths.append(filepath)
            time.sleep(0.1)  # Ensure different modification times
        
        result = glob_files("*.py", temp_dir)
        
        # Should be sorted newest first
        assert len(result) == 3, f"Expected 3 files, got {len(result)}"
        assert "newest_file.py" in result[0], f"Newest file should be first: {result[0]}"
        assert "old_file.py" in result[-1], f"Oldest file should be last: {result[-1]}"
        
        print("✓ Sorting by modification time test passed")


if __name__ == "__main__":
    test_glob_files()
    test_glob_files_advanced()
    test_sorting_behavior()
    print("\nAll glob tests passed! ✅")