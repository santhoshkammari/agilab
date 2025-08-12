
import os
import tempfile
import unittest
from tool_glob import glob_files, glob_files_advanced

import shutil

class TestGeminiBasedToolGlob(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_temp_file(self, name, content=''):
        file_path = os.path.join(self.temp_dir, name)
        with open(file_path, 'w') as f:
            f.write(content)
        self.temp_files.append(file_path)
        return file_path

    def test_glob_with_hidden_files(self):
        """Test that glob does not return hidden files by default."""
        self.create_temp_file(".hidden_file.txt")
        self.create_temp_file("visible_file.txt")
        
        result = glob_files("*.txt", self.temp_dir)
        self.assertEqual(len(result), 1)
        self.assertIn("visible_file.txt", result[0])
        print("✓ test_glob_with_hidden_files passed")

    def test_glob_with_absolute_path(self):
        """Test glob with an absolute path pattern."""
        self.create_temp_file("file1.txt")
        absolute_pattern = os.path.join(self.temp_dir, "*.txt")
        result = glob_files(absolute_pattern)
        self.assertEqual(len(result), 1)
        self.assertIn("file1.txt", result[0])
        print("✓ test_glob_with_absolute_path passed")

    def test_glob_advanced_with_directory_and_file_mix(self):
        """Test glob_files_advanced with a mix of files and directories."""
        self.create_temp_file("file1.txt")
        os.makedirs(os.path.join(self.temp_dir, "subdir1"))
        
        result = glob_files_advanced("*", self.temp_dir, include_dirs=True)
        self.assertEqual(len(result), 2)
        
        result_files_only = glob_files_advanced("*", self.temp_dir, include_dirs=False)
        self.assertEqual(len(result_files_only), 1)
        self.assertIn("file1.txt", result_files_only[0])
        print("✓ test_glob_advanced_with_directory_and_file_mix passed")

    def test_glob_with_empty_directory(self):
        """Test glob on an empty directory."""
        result = glob_files("*", self.temp_dir)
        self.assertEqual(len(result), 0)
        print("✓ test_glob_with_empty_directory passed")

    def test_glob_with_no_matching_files(self):
        """Test glob with a pattern that matches no files."""
        self.create_temp_file("file1.txt")
        result = glob_files("*.log", self.temp_dir)
        self.assertEqual(len(result), 0)
        print("✓ test_glob_with_no_matching_files passed")

if __name__ == '__main__':
    unittest.main()
