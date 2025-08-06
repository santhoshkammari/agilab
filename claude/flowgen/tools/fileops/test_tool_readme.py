import os
import tempfile
from tool_readfile import read_file

def test_read_file():
    """Test the read_file function with various scenarios."""
    
    # Test 1: Basic file reading
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\nLine 2\nLine 3\n")
        temp_file = f.name
    
    try:
        result = read_file(temp_file)
        expected = "    1\tLine 1\n    2\tLine 2\n    3\tLine 3"
        assert result == expected, f"Basic read failed: {result}"
        print("✓ Basic file reading test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 2: File with offset
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        temp_file = f.name
    
    try:
        result = read_file(temp_file, offset=3)
        expected = "    3\tLine 3\n    4\tLine 4\n    5\tLine 5"
        assert result == expected, f"Offset read failed: {result}"
        print("✓ Offset reading test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 3: File with limit
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        temp_file = f.name
    
    try:
        result = read_file(temp_file, limit=2)
        expected = "    1\tLine 1\n    2\tLine 2"
        assert result == expected, f"Limit read failed: {result}"
        print("✓ Limit reading test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 4: File with offset and limit
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        temp_file = f.name
    
    try:
        result = read_file(temp_file, offset=2, limit=2)
        expected = "    2\tLine 2\n    3\tLine 3"
        assert result == expected, f"Offset+limit read failed: {result}"
        print("✓ Offset and limit reading test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 5: Empty file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = f.name
    
    try:
        result = read_file(temp_file)
        expected = "[System Reminder: File exists but has empty contents]"
        assert result == expected, f"Empty file test failed: {result}"
        print("✓ Empty file test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 6: File not found
    try:
        read_file("/nonexistent/file.txt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found test passed")
    
    # Test 7: Relative path validation
    try:
        read_file("relative/path.txt")
        assert False, "Should have raised ValueError for relative path"
    except ValueError as e:
        assert "absolute path" in str(e)
        print("✓ Relative path validation test passed")
    
    # Test 8: Invalid offset
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Line 1\n")
        temp_file = f.name
    
    try:
        try:
            read_file(temp_file, offset=0)
            assert False, "Should have raised ValueError for invalid offset"
        except ValueError as e:
            assert "offset must be >= 1" in str(e)
            print("✓ Invalid offset test passed")
    finally:
        os.unlink(temp_file)
    
    # Test 9: Long line truncation
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        long_line = "x" * 2500
        f.write(f"{long_line}\nShort line\n")
        temp_file = f.name
    
    try:
        result = read_file(temp_file)
        lines = result.split('\n')
        assert lines[0].endswith("..."), "Long line should be truncated"
        assert "Short line" in lines[1], "Short line should be intact"
        print("✓ Long line truncation test passed")
    finally:
        os.unlink(temp_file)
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_read_file()