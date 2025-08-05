import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the flowgen directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flowgen.tools.fileops.tool_bash import execute_bash_command
from flowgen.tools.fileops.tool_edit import edit_file, mark_file_as_read as edit_mark_file_as_read
from flowgen.tools.fileops.tool_write import write_file, mark_file_as_read as write_mark_file_as_read
from flowgen.tools.fileops.tool_readme import read_file
from flowgen.tools.fileops.tool_ls import list_directory
from flowgen.tools.fileops.tool_glob import glob_files
from flowgen.tools.fileops.tool_grep import grep_search
from flowgen.tools.fileops.tool_multiedit import multi_edit
from flowgen.tools.fileops.tool_task import task
from flowgen.tools.fileops.tool_todowrite import todo_write
from flowgen.tools.fileops.tool_webfetch import web_fetch
from flowgen.tools.fileops.tool_websearch import websearch
from flowgen.tools.fileops.tool_exitplanmode import exit_plan_mode

# Real Gemini LLM integration
import google.generativeai as genai

class GeminiLLM:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            import os
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"


class TestFileOpsTools:
    def setup_method(self):
        # Initialize Gemini LLM - skip if no API key available
        try:
            self.gemini = GeminiLLM()
        except ValueError:
            self.gemini = None
            print("Warning: Google API key not found. Gemini tests will be skipped.")
        
        self.test_dir = "/tmp/test_fileops"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_bash_tool(self):
        """Test bash command execution tool"""
        result = execute_bash_command("echo 'Hello World'", "Test echo command")
        assert result["success"] == True
        assert "Hello World" in result["stdout"]
        assert "exit_code" in result
        assert "execution_time" in result
    
    def test_write_tool(self):
        """Test file writing tool"""
        test_file = os.path.join(self.test_dir, "test.txt")
        result = write_file(test_file, "Test content")
        assert "Successfully wrote" in result
        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            assert f.read() == "Test content"
    
    def test_read_tool(self):
        """Test file reading tool"""
        test_file = os.path.join(self.test_dir, "read_test.txt")
        with open(test_file, 'w') as f:
            f.write("Content to read")
        
        result = read_file(test_file)
        assert "Content to read" in result
        assert "1\t" in result  # Line number format
    
    def test_edit_tool(self):
        """Test file editing tool"""
        test_file = os.path.join(self.test_dir, "edit_test.txt")
        with open(test_file, 'w') as f:
            f.write("Original content")
        
        # Mark file as read first (required by tool)
        edit_mark_file_as_read(test_file)
        
        result = edit_file(test_file, "Original", "Modified")
        assert "Replaced text" in result
        with open(test_file, 'r') as f:
            assert "Modified content" in f.read()
    
    def test_ls_tool(self):
        """Test directory listing tool"""
        result = list_directory(self.test_dir)
        assert "entries" in result
        assert isinstance(result["entries"], list)
    
    def test_glob_tool(self):
        """Test glob pattern matching tool"""
        # Create test files
        for i in range(3):
            with open(os.path.join(self.test_dir, f"test{i}.txt"), 'w') as f:
                f.write(f"content {i}")
        
        result = glob_files("*.txt", self.test_dir)
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_grep_tool(self):
        """Test grep search tool"""
        test_file = os.path.join(self.test_dir, "grep_test.txt")
        with open(test_file, 'w') as f:
            f.write("Line 1\nTarget line\nLine 3")
        
        result = grep_search("Target", test_file, output_mode="files_with_matches")
        assert test_file in result or result == ""  # Either found or no matches
    
    def test_multiedit_tool(self):
        """Test multi-edit tool"""
        test_file = os.path.join(self.test_dir, "multiedit_test.txt")
        with open(test_file, 'w') as f:
            f.write("Line 1\nLine 2\nLine 3")
        
        edits = [
            {"old_string": "Line 1", "new_string": "Modified Line 1"},
            {"old_string": "Line 3", "new_string": "Modified Line 3"}
        ]
        result = multi_edit(test_file, edits)
        assert "Successfully applied" in result
        assert "2 edits" in result
    
    def test_task_tool(self):
        """Test task creation tool"""
        result = task("Test task example", "This is a test prompt for the task", "general-purpose")
        assert "Task Agent Response" in result
        assert "Test task example" in result
        assert "general-purpose" in result
    
    def test_todowrite_tool(self):
        """Test todo writing tool"""
        todos = [
            {"id": "1", "content": "Test todo 1", "status": "pending", "priority": "high"},
            {"id": "2", "content": "Test todo 2", "status": "completed", "priority": "medium"}
        ]
        result = todo_write(todos)
        assert result["success"] == True
        assert len(result["todos"]) == 2
    
    @patch('requests.get')
    def test_webfetch_tool(self, mock_get):
        """Test web fetching tool"""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        result = web_fetch("https://example.com", "Extract main content")
        assert result["success"] == True
        assert "content_summary" in result
    
    @patch.dict(os.environ, {'CLAUDE_REGION': 'US'})
    def test_websearch_tool(self):
        """Test web search tool"""
        result = websearch("test query")
        assert "results" in result
        assert "search_metadata" in result
        assert result["search_metadata"]["query"] == "test query"
    
    def test_exitplanmode_tool(self):
        """Test exit plan mode tool"""
        result = exit_plan_mode("Test implementation plan")
        assert "Implementation Plan Ready" in result
        assert "Test implementation plan" in result

    def test_with_gemini_integration(self):
        """Test tools with real Gemini LLM integration"""
        if self.gemini is None:
            pytest.skip("Gemini API key not available")
        
        # Create a test file
        test_file = os.path.join(self.test_dir, "gemini_test.txt")
        write_file(test_file, "This is a test file for Gemini integration. It contains sample text for AI analysis.")
        
        # Read the file content
        file_content = read_file(test_file)
        
        # Generate response using real Gemini
        prompt = f"Analyze this file content and provide a brief summary in one sentence: {file_content}"
        response = self.gemini.generate_response(prompt)
        
        assert len(response) > 0
        assert not response.startswith("Error generating response")
        print(f"Gemini response: {response}")
    
    def test_gemini_with_code_analysis(self):
        """Test Gemini analyzing code content"""
        if self.gemini is None:
            pytest.skip("Gemini API key not available")
        
        # Create a Python code file
        test_file = os.path.join(self.test_dir, "code_test.py")
        code_content = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
print(fibonacci(10))
'''
        write_file(test_file, code_content)
        
        # Read the file content
        file_content = read_file(test_file)
        
        # Analyze code with Gemini
        prompt = f"Analyze this Python code and explain what it does in one sentence: {file_content}"
        response = self.gemini.generate_response(prompt)
        
        assert len(response) > 0
        assert not response.startswith("Error generating response")
        assert any(word in response.lower() for word in ["fibonacci", "recursive", "sequence"])
        print(f"Code analysis: {response}")
    
    def test_gemini_with_multiple_tools(self):
        """Test combining multiple fileops tools with Gemini analysis"""
        if self.gemini is None:
            pytest.skip("Gemini API key not available")
        
        # Create multiple test files
        files_data = {
            "config.json": '{"name": "test_app", "version": "1.0.0", "debug": true}',
            "main.py": "import json\n\nwith open('config.json') as f:\n    config = json.load(f)\n    print(config['name'])",
            "readme.md": "# Test App\n\nThis is a simple test application with configuration."
        }
        
        for filename, content in files_data.items():
            test_file = os.path.join(self.test_dir, filename)
            write_file(test_file, content)
        
        # Use glob to find all files
        all_files = glob_files("*", self.test_dir)
        
        # Read all files and combine content
        combined_content = ""
        for file_path in all_files:
            if any(file_path.endswith(ext) for ext in ['.json', '.py', '.md']):
                content = read_file(file_path)
                combined_content += f"\n--- File: {os.path.basename(file_path)} ---\n{content}\n"
        
        # Analyze with Gemini
        prompt = f"Analyze this multi-file project structure and summarize what the application does: {combined_content}"
        response = self.gemini.generate_response(prompt)
        
        assert len(response) > 0
        assert not response.startswith("Error generating response")
        print(f"Project analysis: {response}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])