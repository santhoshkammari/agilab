from fastmcp import FastMCP
import os
from typing import Optional, List, Dict, Any

def read_file_with_line_numbers(file_path: str) -> str:
    """Read file content and return with line numbers"""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist"
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Format with line numbers starting from 1
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            # Add line number followed by content, preserving line breaks
            numbered_lines.append(f"{i:3d}→{line}")
        
        return ''.join(numbered_lines)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

def replace_str_in_file(file_path: str, edits: List[Dict[str, Any]]) -> str:
    """
    Replace multiple sections in a file with new content.
    
    Args:
        file_path: Path to the file
        edits: List of dictionaries, each containing:
            - line_start: Starting line number (1-indexed)
            - line_end: Ending line number (1-indexed, inclusive)
            - new_string: New content to replace with
    
    Edits are applied from bottom to top to maintain line number validity.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist"
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file"
    
    # Validate edits
    for i, edit in enumerate(edits):
        if 'line_start' not in edit or 'line_end' not in edit or 'new_string' not in edit:
            return f"Error: Edit {i+1} missing required fields (line_start, line_end, new_string)"
        
        line_start = edit['line_start']
        line_end = edit['line_end']
        
        if line_start < 1 or line_end < line_start:
            return f"Error: Edit {i+1}: line_start must be >= 1 and line_end must be >= line_start"
    
    # Sort edits by line_start in descending order (bottom to top)
    sorted_edits = sorted(edits, key=lambda x: x['line_start'], reverse=True)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = []
        
        # Apply each edit from bottom to top
        for edit in sorted_edits:
            line_start = edit['line_start']
            line_end = edit['line_end']
            new_string = edit['new_string']
            
            # Adjust to 0-based indexing
            start_idx = line_start - 1
            end_idx = line_end - 1
            
            # Check if line numbers are within bounds
            if start_idx >= len(lines) or end_idx >= len(lines):
                return f"Error: Lines {line_start}-{line_end} out of range. File has {len(lines)} lines."
            
            # Split new_string into lines if it contains multiple lines
            new_lines = new_string.splitlines(keepends=True)
            if new_string and not new_string.endswith('\n') and new_lines:
                new_lines[-1] += '\n'
            
            # Replace the lines
            lines = lines[:start_idx] + new_lines + lines[end_idx+1:]
            
            results.append(f"Replaced lines {line_start}-{line_end}")
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return f"Successfully applied {len(edits)} edit(s) to '{file_path}':\n" + "\n".join(results)
    
    except Exception as e:
        return f"Error replacing content in file '{file_path}': {str(e)}"

# Create FastMCP server
mcp = FastMCP("File Tools Optimized Server")

@mcp.tool
def read_file(file_path: str) -> str:
    """Read file content and return with line numbers starting from index 1."""
    return read_file_with_line_numbers(file_path)

@mcp.tool
def replace_str(file_path: str, edits: List[Dict[str, Any]]) -> str:
    """
    Replace multiple sections in a file with new content in one operation.
    
    Args:
        file_path: Path to the file to edit
        edits: List of edit operations, each containing:
            - line_start: Starting line number (1-indexed)
            - line_end: Ending line number (1-indexed, inclusive)
            - new_string: New content to replace with
    
    Example:
        edits = [
            {"line_start": 5, "line_end": 7, "new_string": "new content for lines 5-7\\n"},
            {"line_start": 10, "line_end": 10, "new_string": "replacement for line 10\\n"}
        ]
    """
    return replace_str_in_file(file_path, edits)

tool_functions = {
    "read_file": read_file,
    "replace_str": replace_str,
}

if __name__ == "__main__":
    mcp.run()