import os
import tempfile
import time
from tool_bash import execute_bash_command, format_command_output, bash_tool_wrapper


def test_execute_bash_command():
    """Test the execute_bash_command function with various scenarios."""
    
    # Test 1: Simple successful command
    result = execute_bash_command("echo 'Hello World'")
    assert result['success'] == True
    assert "Hello World" in result['stdout']
    assert result['exit_code'] == 0
    assert result['execution_time'] > 0
    assert result['truncated'] == False
    print("✓ Simple command execution test passed")
    
    # Test 2: Command with description
    result = execute_bash_command("pwd", description="Show current directory")
    assert result['success'] == True
    assert len(result['stdout']) > 0
    print("✓ Command with description test passed")
    
    # Test 3: Command that fails
    result = execute_bash_command("nonexistent-command-xyz")
    assert result['success'] == False
    assert result['exit_code'] != 0
    assert len(result['stderr']) > 0
    print("✓ Failed command test passed")
    
    # Test 4: Command with custom timeout
    result = execute_bash_command("sleep 0.1", timeout=1000)  # 1 second timeout
    assert result['success'] == True
    print("✓ Custom timeout test passed")
    
    # Test 5: Empty command validation
    try:
        execute_bash_command("")
        assert False, "Should have raised ValueError for empty command"
    except ValueError as e:
        assert "cannot be empty" in str(e)
        print("✓ Empty command validation test passed")
    
    # Test 6: Invalid timeout validation
    try:
        execute_bash_command("echo test", timeout=700000)  # Over 10 minutes
        assert False, "Should have raised ValueError for invalid timeout"
    except ValueError as e:
        assert "Timeout must be between" in str(e)
        print("✓ Invalid timeout validation test passed")
    
    # Test 7: Prohibited command validation (find)
    try:
        execute_bash_command("find . -name '*.py'")
        assert False, "Should have raised ValueError for prohibited find command"
    except ValueError as e:
        assert "Use Glob tool" in str(e)
        print("✓ Prohibited find command validation test passed")
    
    # Test 8: Prohibited command validation (grep)
    try:
        execute_bash_command("grep pattern file.txt")
        assert False, "Should have raised ValueError for prohibited grep command"
    except ValueError as e:
        assert "Use Grep tool" in str(e)
        print("✓ Prohibited grep command validation test passed")
    
    # Test 9: Prohibited command validation (cat)
    try:
        execute_bash_command("cat file.txt")
        assert False, "Should have raised ValueError for prohibited cat command"
    except ValueError as e:
        assert "Use Read tool" in str(e)
        print("✓ Prohibited cat command validation test passed")
    
    # Test 10: Interactive command validation
    try:
        execute_bash_command("git rebase -i HEAD~3")
        assert False, "Should have raised ValueError for interactive command"
    except ValueError as e:
        assert "Interactive commands" in str(e)
        print("✓ Interactive command validation test passed")
    
    # Test 11: Command with output
    result = execute_bash_command("echo -e 'Line 1\\nLine 2\\nLine 3'")
    assert result['success'] == True
    assert "Line 1" in result['stdout']
    assert "Line 2" in result['stdout']
    assert "Line 3" in result['stdout']
    print("✓ Multi-line output test passed")
    
    # Test 12: Command with stderr
    result = execute_bash_command("echo 'error message' >&2; echo 'stdout message'")
    assert result['success'] == True
    assert "stdout message" in result['stdout']
    assert "error message" in result['stderr']
    print("✓ stderr output test passed")


def test_timeout_functionality():
    """Test timeout functionality with a slow command."""
    print("Testing timeout functionality...")
    
    try:
        # This should timeout
        execute_bash_command("sleep 2", timeout=500)  # 0.5 second timeout
        assert False, "Should have raised TimeoutError"
    except TimeoutError as e:
        assert "timed out" in str(e)
        print("✓ Timeout functionality test passed")


def test_format_command_output():
    """Test the format_command_output function."""
    
    # Test 1: Successful command formatting
    result = {
        'success': True,
        'stdout': 'Hello World',
        'stderr': '',
        'exit_code': 0,
        'execution_time': 0.123,
        'truncated': False
    }
    
    formatted = format_command_output(result, "echo 'Hello World'", "Print greeting")
    assert "Command: Print greeting" in formatted
    assert "$ echo 'Hello World'" in formatted
    assert "Hello World" in formatted
    assert "[Execution time: 0.12s]" in formatted
    print("✓ Successful command formatting test passed")
    
    # Test 2: Failed command formatting
    result = {
        'success': False,
        'stdout': '',
        'stderr': 'command not found',
        'exit_code': 127,
        'execution_time': 0.056,
        'truncated': False
    }
    
    formatted = format_command_output(result, "badcommand")
    assert "[ERROR] Command failed with exit code 127" in formatted
    assert "command not found" in formatted
    print("✓ Failed command formatting test passed")
    
    # Test 3: Truncated output formatting
    result = {
        'success': True,
        'stdout': 'Some output',
        'stderr': '',
        'exit_code': 0,
        'execution_time': 0.1,
        'truncated': True
    }
    
    formatted = format_command_output(result, "somecommand")
    assert "[Note: Output was truncated due to size]" in formatted
    print("✓ Truncated output formatting test passed")


def test_bash_tool_wrapper():
    """Test the main wrapper function."""
    
    # Test 1: Simple wrapper usage
    output = bash_tool_wrapper("echo 'test'", "Echo test message")
    assert "Command: Echo test message" in output
    assert "$ echo 'test'" in output
    assert "test" in output
    print("✓ Bash tool wrapper test passed")
    
    # Test 2: Wrapper with timeout
    output = bash_tool_wrapper("echo 'timeout test'", timeout=5000)
    assert "timeout test" in output
    print("✓ Wrapper with timeout test passed")


def test_file_operations():
    """Test file operations with temporary files."""
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content for bash tool")
        temp_file = f.name
    
    try:
        # Test creating a directory and moving the file
        temp_dir = tempfile.mkdtemp()
        result = execute_bash_command(f'mkdir -p "{temp_dir}/subdir" && mv "{temp_file}" "{temp_dir}/subdir/"')
        assert result['success'] == True
        
        # Verify the file was moved
        moved_file = os.path.join(temp_dir, "subdir", os.path.basename(temp_file))
        assert os.path.exists(moved_file)
        print("✓ File operations test passed")
        
        # Clean up
        os.unlink(moved_file)
        os.rmdir(os.path.join(temp_dir, "subdir"))
        os.rmdir(temp_dir)
        
    except Exception:
        # Clean up in case of failure
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        raise


def test_command_chaining():
    """Test command chaining with && and ;"""
    
    # Test 1: Successful command chain with &&
    result = execute_bash_command("echo 'first' && echo 'second'")
    assert result['success'] == True
    assert "first" in result['stdout']
    assert "second" in result['stdout']
    print("✓ Command chaining with && test passed")
    
    # Test 2: Command chain with ; (always executes second)
    result = execute_bash_command("echo 'first'; echo 'second'")
    assert result['success'] == True
    assert "first" in result['stdout']
    assert "second" in result['stdout']
    print("✓ Command chaining with ; test passed")
    
    # Test 3: Failed first command with &&
    result = execute_bash_command("false && echo 'should not see this'")
    assert result['success'] == False
    assert "should not see this" not in result['stdout']
    print("✓ Failed command chain test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running Bash tool tests...\n")
    
    test_execute_bash_command()
    test_timeout_functionality()
    test_format_command_output()
    test_bash_tool_wrapper()
    test_file_operations()
    test_command_chaining()
    
    print("\nAll Bash tool tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()