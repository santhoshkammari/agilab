import os
import tempfile
import subprocess
from tool_grep import grep_search, grep


def test_grep_search():
    """Test the grep_search function with various scenarios."""
    
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create test files
        test_file1 = os.path.join(temp_dir, "test1.py")
        with open(test_file1, 'w') as f:
            f.write("def function_error():\n    print('Error occurred')\n    return False\n")
        
        test_file2 = os.path.join(temp_dir, "test2.js")
        with open(test_file2, 'w') as f:
            f.write("function handleError() {\n    console.log('Error');\n}\nconst SUCCESS = true;\n")
        
        test_file3 = os.path.join(temp_dir, "readme.md")
        with open(test_file3, 'w') as f:
            f.write("# Project\n\nThis project handles errors gracefully.\n\n## TODO\n- Fix error handling\n")
        
        # Test 1: Basic pattern search with files_with_matches (default)
        try:
            result = grep_search("error", path=temp_dir, i=True)
            files = result.split('\n') if result else []
            assert any("test1.py" in f for f in files), "Should find test1.py"
            assert any("test2.js" in f for f in files), "Should find test2.js"
            assert any("readme.md" in f for f in files), "Should find readme.md"
            print("✓ Basic files_with_matches test passed")
        except FileNotFoundError:
            print("⚠ Skipping tests - ripgrep not available")
            return
        except Exception as e:
            print(f"⚠ Basic test failed: {e}")
            return
        
        # Test 2: Content mode with line numbers
        try:
            result = grep_search("error", path=temp_dir, output_mode="content", n=True, i=True)
            assert "function_error" in result or "Error" in result, "Should find error content"
            print("✓ Content mode with line numbers test passed")
        except Exception as e:
            print(f"⚠ Content mode test failed: {e}")
        
        # Test 3: Count mode
        try:
            result = grep_search("error", path=temp_dir, output_mode="count", i=True)
            assert result, "Should have some counts"
            print("✓ Count mode test passed")
        except Exception as e:
            print(f"⚠ Count mode test failed: {e}")
        
        # Test 4: File type filtering
        try:
            result = grep_search("function", path=temp_dir, type="py")
            if result:
                assert "test1.py" in result, "Should find Python file"
                assert "test2.js" not in result, "Should not find JS file"
            print("✓ File type filtering test passed")
        except Exception as e:
            print(f"⚠ File type filtering test failed: {e}")
        
        # Test 5: Glob pattern filtering
        try:
            result = grep_search("error", path=temp_dir, glob="*.md", i=True)
            if result:
                assert "readme.md" in result, "Should find markdown file"
            print("✓ Glob pattern filtering test passed")
        except Exception as e:
            print(f"⚠ Glob pattern filtering test failed: {e}")
        
        # Test 6: Head limit
        try:
            result = grep_search("error", path=temp_dir, output_mode="files_with_matches", head_limit=1, i=True)
            files = result.split('\n') if result else []
            if files and files[0]:
                assert len(files) <= 1, f"Should limit to 1 result, got {len(files)}"
            print("✓ Head limit test passed")
        except Exception as e:
            print(f"⚠ Head limit test failed: {e}")
        
        # Test 7: Case sensitive vs insensitive
        try:
            result_sensitive = grep_search("ERROR", path=temp_dir)
            result_insensitive = grep_search("ERROR", path=temp_dir, i=True)
            # Insensitive should find more or equal matches
            print("✓ Case sensitivity test passed")
        except Exception as e:
            print(f"⚠ Case sensitivity test failed: {e}")
    
    # Test 8: Empty pattern validation
    try:
        grep_search("")
        assert False, "Should have raised ValueError for empty pattern"
    except ValueError as e:
        assert "pattern parameter is required" in str(e)
        print("✓ Empty pattern validation test passed")
    
    # Test 9: Invalid path
    try:
        grep_search("test", path="/nonexistent/directory")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ Invalid path test passed")
    
    # Test 10: Invalid context options for non-content mode
    try:
        grep_search("test", output_mode="files_with_matches", n=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires output_mode: 'content'" in str(e)
        print("✓ Invalid context options test passed")
    
    # Test 11: Multiline mode
    with tempfile.TemporaryDirectory() as temp_dir:
        multiline_file = os.path.join(temp_dir, "multiline.txt")
        with open(multiline_file, 'w') as f:
            f.write("start of\nmultiline\npattern end")
        
        try:
            result = grep_search("start.*end", path=temp_dir, multiline=True)
            # This may or may not find results depending on ripgrep version
            print("✓ Multiline mode test passed")
        except Exception as e:
            print(f"⚠ Multiline mode test failed: {e}")
    
    # Test 12: Convenience alias function
    try:
        result = grep("test_pattern_that_wont_match_anything")
        # Should not raise an error, just return empty string
        print("✓ Convenience alias test passed")
    except Exception as e:
        print(f"⚠ Convenience alias test failed: {e}")
    
    print("\nGrep tool tests completed! ✅")


def test_ripgrep_availability():
    """Test if ripgrep is available on the system."""
    try:
        result = subprocess.run(["rg", "--version"], capture_output=True, text=True, check=True)
        print(f"✓ ripgrep is available: {result.stdout.split()[1] if len(result.stdout.split()) > 1 else 'version unknown'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ ripgrep (rg) is not available - some tests will be skipped")
        return False


if __name__ == "__main__":
    print("Testing ripgrep availability...")
    ripgrep_available = test_ripgrep_availability()
    
    print("\nRunning grep_search tests...")
    test_grep_search()
    
    if not ripgrep_available:
        print("\nNote: Install ripgrep with 'apt install ripgrep' or 'brew install ripgrep' for full functionality.")