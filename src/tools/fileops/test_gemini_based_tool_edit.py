
import os
import tempfile
import unittest
from tool_edit import edit_file, mark_file_as_read, clear_read_files

class TestGeminiBasedToolEdit(unittest.TestCase):

    def setUp(self):
        clear_read_files()
        self.temp_files = []

    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def create_temp_file(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            self.temp_files.append(f.name)
            return f.name

    def test_edit_with_special_characters(self):
        """Test editing content with special characters and unicode."""
        temp_file = self.create_temp_file("Hello, 世界! 🌍\nSpecial chars: àáâãäå")
        mark_file_as_read(temp_file)
        
        edit_file(temp_file, "àáâãäå", "aeiou")
        
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("aeiou", content)
        self.assertNotIn("àáâãäå", content)
        print("✓ test_edit_with_special_characters passed")

    def test_edit_large_file(self):
        """Test editing a large file."""
        content = "A" * 10000 + "\n" + "B" * 5000 + "\n" + "C" * 5000
        temp_file = self.create_temp_file(content)
        mark_file_as_read(temp_file)
        
        edit_file(temp_file, "B" * 5000, "D" * 5000)
        
        with open(temp_file, 'r') as f:
            new_content = f.read()
        self.assertIn("D" * 5000, new_content)
        self.assertNotIn("B" * 5000, new_content)
        print("✓ test_edit_large_file passed")

    def test_no_change_on_identical_strings(self):
        """Test that no change is made when old and new strings are identical."""
        temp_file = self.create_temp_file("Hello world")
        mark_file_as_read(temp_file)
        
        with self.assertRaises(ValueError):
            edit_file(temp_file, "Hello", "Hello")
        print("✓ test_no_change_on_identical_strings passed")

    def test_edit_file_with_no_trailing_newline(self):
        """Test editing a file that doesn't end with a newline."""
        temp_file = self.create_temp_file("Hello world")
        mark_file_as_read(temp_file)
        
        edit_file(temp_file, "world", "Gemini")
        
        with open(temp_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "Hello Gemini")
        print("✓ test_edit_file_with_no_trailing_newline passed")

    def test_replace_all_with_overlapping_matches(self):
        """Test replace_all with overlapping matches."""
        temp_file = self.create_temp_file("ababab")
        mark_file_as_read(temp_file)
        
        edit_file(temp_file, "aba", "c", replace_all=True)
        
        with open(temp_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "cbab")
        print("✓ test_replace_all_with_overlapping_matches passed")

if __name__ == '__main__':
    unittest.main()
