import os
import tempfile
import unittest
from tool_grep import grep_search

class TestGeminiBasedToolGrep(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []

    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def create_temp_file(self, name, content=''):
        file_path = os.path.join(self.temp_dir, name)
        with open(file_path, 'w') as f:
            f.write(content)
        self.temp_files.append(file_path)
        return file_path

    def test_grep_with_context(self):
        """Test grep with before and after context."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        temp_file = self.create_temp_file("context.txt", content)
        
        result = grep_search("Line 3", path=temp_file, output_mode="content", A=1, B=1)
        self.assertIn("Line 2", result)
        self.assertIn("Line 3", result)
        self.assertIn("Line 4", result)
        self.assertNotIn("Line 1", result)
        self.assertNotIn("Line 5", result)
        print("✓ test_grep_with_context passed")

    def test_grep_with_multiline_pattern(self):
        """Test grep with a multiline pattern."""
        content = "Hello\nWorld\nThis is a test"
        temp_file = self.create_temp_file("multiline.txt", content)
        
        result = grep_search("Hello\nWorld", path=temp_file, multiline=True, output_mode='content')
        self.assertIn("Hello\nWorld", result)
        print("✓ test_grep_with_multiline_pattern passed")

    def test_grep_with_word_boundary(self):
        """Test grep with word boundary matching."""
        content = "This is a test, testing the tester."
        temp_file = self.create_temp_file("word_boundary.txt", content)
        
        result = grep_search(r'test', path=temp_file, output_mode='content', w=True)
        self.assertIn("This is a test", result)
        self.assertNotIn("testing", result)
        self.assertNotIn("tester", result)
        print("✓ test_grep_with_word_boundary passed")

    def test_grep_with_invert_match(self):
        """Test grep with invert match."""
        content = "Line 1\nLine 2\nLine 3"
        temp_file = self.create_temp_file("invert.txt", content)
        
        result = grep_search("Line 2", path=temp_file, output_mode="content", v=True)
        self.assertIn("Line 1", result)
        self.assertNotIn("Line 2", result)
        self.assertIn("Line 3", result)
        print("✓ test_grep_with_invert_match passed")

    def test_grep_with_file_and_line_number(self):
        """Test grep with file and line number."""
        content = "Line 1\nLine 2\nLine 3"
        temp_file = self.create_temp_file("file_line.txt", content)
        
        result = grep_search("Line 2", path=temp_file, output_mode="content", n=True)
        self.assertIn("2:Line 2", result)
        print("✓ test_grep_with_file_and_line_number passed")

if __name__ == '__main__':
    unittest.main()
