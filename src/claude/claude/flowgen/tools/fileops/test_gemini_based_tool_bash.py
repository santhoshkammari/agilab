
import os
import tempfile
import unittest
from tool_bash import execute_bash_command, bash_tool_wrapper

class TestGeminiBasedToolBash(unittest.TestCase):

    def test_complex_command_chains(self):
        """Test complex command chains with pipes and redirections."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
        
        command = f"echo 'Line 1\nLine 2\nLine 3' | grep 'Line 2' > {temp_file}"
        result = execute_bash_command(command)
        self.assertTrue(result['success'])
        
        with open(temp_file, 'r') as f:
            content = f.read()
        self.assertIn('Line 2', content)
        os.unlink(temp_file)
        print("✓ test_complex_command_chains passed")

    def test_environment_variable_usage(self):
        """Test the usage of environment variables in commands."""
        os.environ['GEMINI_TEST_VAR'] = 'Hello from Gemini'
        result = execute_bash_command("echo $GEMINI_TEST_VAR")
        self.assertTrue(result['success'])
        self.assertIn('Hello from Gemini', result['stdout'])
        del os.environ['GEMINI_TEST_VAR']
        print("✓ test_environment_variable_usage passed")

    def test_large_output_truncation(self):
        """Test that large outputs are correctly truncated."""
        # This command generates 40000 characters ('a' * 40000)
        command = "python3 -c \"print('a' * 40000)\""
        result = execute_bash_command(command)
        self.assertTrue(result['success'])
        self.assertTrue(result['truncated'])
        self.assertIn('[OUTPUT TRUNCATED]', result['stdout'])
        self.assertLess(len(result['stdout']), 35000)
        print("✓ test_large_output_truncation passed")

    def test_command_in_subdirectory(self):
        """Test executing a command within a subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix='.txt') as f:
                temp_file_name = os.path.basename(f.name)

            command = f"cd {temp_dir} && ls {temp_file_name}"
            result = execute_bash_command(command)
            self.assertTrue(result['success'])
            self.assertIn(temp_file_name, result['stdout'])
        print("✓ test_command_in_subdirectory passed")
        
    def test_stderr_and_exit_code(self):
        """Test a command that writes to stderr and exits with a non-zero code."""
        command = "echo 'error message' >&2; exit 1"
        result = execute_bash_command(command)
        self.assertFalse(result['success'])
        self.assertEqual(result['exit_code'], 1)
        self.assertIn('error message', result['stderr'])
        print("✓ test_stderr_and_exit_code passed")

    def test_wrapper_function(self):
        """Test the bash_tool_wrapper function."""
        output = bash_tool_wrapper("echo 'wrapper test'")
        self.assertIn("$ echo 'wrapper test'", output)
        self.assertIn("wrapper test", output)
        print("✓ test_wrapper_function passed")

if __name__ == '__main__':
    unittest.main()
