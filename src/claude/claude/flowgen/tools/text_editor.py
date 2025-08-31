import os
import json


SYSTEM_PROMPT = """You are a text editing assistant. You have access to the following tools for file operations:

- text_editor_str_replace: Replace a specific string in a file with a new string (must be unique)
- text_editor_create: Create a new file with specified content
- text_editor_insert: Insert text at a specific line number in a file
- text_editor_copy_paste: Copy lines from source file and paste into destination file
- text_editor_view: View file contents with optional line range

These tools allow you to perform precise file editing operations. Use them to create, modify, and manage text files."""


def text_editor_str_replace(path: str, old_str: str, new_str: str):
    """Replace a specific string in a file with a new string"""
    try:
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_str not in content:
            return {"success": False, "error": f"String not found in file: {old_str[:50]}..."}
        
        # Count occurrences
        count = content.count(old_str)
        if count > 1:
            return {"success": False, "error": f"Multiple occurrences ({count}) found. String must be unique."}
        
        # Perform replacement
        new_content = content.replace(old_str, new_str)
        
        # Write new content
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            "success": True,
            "message": f"Successfully replaced text in {path}",
            "changes": 1
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def text_editor_create(path: str, file_text: str):
    """Create a new file with specified content"""
    try:
        if os.path.exists(path):
            return {"success": False, "error": f"File already exists: {path}"}
        
        # Create directory if it doesn't exist (only if path contains directory)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(file_text)
        
        return {
            "success": True,
            "message": f"Successfully created file: {path}",
            "lines": len(file_text.splitlines())
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def text_editor_insert(path: str, insert_line: int, new_str: str):
    """Insert text at a specific line in a file"""
    try:
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Insert at specified line (0 = beginning)
        if insert_line == 0:
            lines.insert(0, new_str + '\n')
        elif insert_line <= len(lines):
            lines.insert(insert_line, new_str + '\n')
        else:
            return {"success": False, "error": f"Line number {insert_line} is beyond file length ({len(lines)})"}
        
        # Write new content
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return {
            "success": True,
            "message": f"Successfully inserted text at line {insert_line} in {path}",
            "new_line_count": len(lines)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def text_editor_copy_paste(source_path: str, source_start: int, source_end: int, 
                          dest_path: str, dest_start: int, dest_end: int):
    """Copy lines from source file and paste into destination file"""
    try:
        # Read source file
        if not os.path.exists(source_path):
            return {"success": False, "error": f"Source file not found: {source_path}"}
        
        with open(source_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        
        # Validate source range
        if source_start < 1 or source_end > len(source_lines) or source_start > source_end:
            return {"success": False, "error": f"Invalid source range: {source_start}-{source_end} (file has {len(source_lines)} lines)"}
        
        # Extract source content (convert to 0-based indexing)
        copied_lines = source_lines[source_start-1:source_end]
        
        # Read destination file (create if doesn't exist)
        dest_lines = []
        if os.path.exists(dest_path):
            with open(dest_path, 'r', encoding='utf-8') as f:
                dest_lines = f.readlines()
        else:
            # Create directory if it doesn't exist (only if path contains directory)
            directory = os.path.dirname(dest_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        
        # Validate destination range
        if dest_start < 1:
            return {"success": False, "error": f"Invalid destination start line: {dest_start}"}
        
        # If dest_end is beyond file length, extend the file
        while len(dest_lines) < dest_end:
            dest_lines.append('\n')
        
        # Replace lines in destination (convert to 0-based indexing)
        dest_lines[dest_start-1:dest_end] = copied_lines
        
        # Write destination file
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.writelines(dest_lines)
        
        return {
            "success": True,
            "message": f"Successfully copied {len(copied_lines)} lines from {source_path}[{source_start}:{source_end}] to {dest_path}[{dest_start}:{dest_end}]",
            "copied_lines": len(copied_lines),
            "source_range": f"{source_start}-{source_end}",
            "dest_range": f"{dest_start}-{dest_end}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def text_editor_view(path: str, start_line: int = 1, end_line: int = None):
    """View contents of a file with optional line range"""
    try:
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Handle line range
        if end_line is None:
            end_line = total_lines
        
        if start_line < 1 or start_line > total_lines:
            return {"success": False, "error": f"Invalid start line: {start_line} (file has {total_lines} lines)"}
        
        if end_line < start_line or end_line > total_lines:
            end_line = total_lines
        
        # Extract requested lines (convert to 0-based indexing)
        requested_lines = lines[start_line-1:end_line]
        
        # Format with line numbers
        formatted_lines = []
        for i, line in enumerate(requested_lines, start=start_line):
            formatted_lines.append(f"{i:4d}→{line.rstrip()}")
        
        return {
            "success": True,
            "content": "\n".join(formatted_lines),
            "total_lines": total_lines,
            "showing_range": f"{start_line}-{start_line + len(requested_lines) - 1}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Compatibility functions for str_replace_editor interface
def str_replace_based_edit_tool(command: str, **kwargs):
    """String replacement based editing tool compatible with str_replace_editor interface"""
    if command == "str_replace":
        return text_editor_str_replace(
            path=kwargs.get("path"),
            old_str=kwargs.get("old_str"),
            new_str=kwargs.get("new_str")
        )
    elif command == "create":
        return text_editor_create(
            path=kwargs.get("path"),
            file_text=kwargs.get("file_text", "")
        )
    elif command == "insert":
        return text_editor_insert(
            path=kwargs.get("path"),
            insert_line=kwargs.get("insert_line", 0),
            new_str=kwargs.get("new_str", "")
        )
    elif command == "view":
        return text_editor_view(
            path=kwargs.get("path"),
            start_line=kwargs.get("start_line", 1),
            end_line=kwargs.get("end_line")
        )
    else:
        return {"success": False, "error": f"Unknown command: {command}"}


tool_functions = {
    "text_editor_str_replace": text_editor_str_replace,
    "text_editor_create": text_editor_create,
    "text_editor_insert": text_editor_insert,
    "text_editor_copy_paste": text_editor_copy_paste,
    "text_editor_view": text_editor_view,
    "str_replace_based_edit_tool": str_replace_based_edit_tool,
}


def run_example():
    """Example usage of text editor tools"""
    # Test create
    result = text_editor_create("test_file.txt", "Hello\nWorld\nThis is a test")
    print("Create result:", result)
    
    # Test view
    result = text_editor_view("test_file.txt")
    print("View result:", result)
    
    # Test str_replace
    result = text_editor_str_replace("test_file.txt", "World", "Python")
    print("Replace result:", result)
    
    # Test insert
    result = text_editor_insert("test_file.txt", 1, "First line inserted")
    print("Insert result:", result)
    
    # Test copy_paste
    result = text_editor_copy_paste("test_file.txt", 1, 2, "test_copy.txt", 1, 1)
    print("Copy paste result:", result)
    
    # Clean up
    if os.path.exists("test_file.txt"):
        os.remove("test_file.txt")
    if os.path.exists("test_copy.txt"):
        os.remove("test_copy.txt")


if __name__ == '__main__':
    run_example()