from fastmcp import FastMCP
import os
from typing import Optional


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


def replace_str_in_file(file_path: str, line_start: int, line_end: int, new_string: str) -> str:
    """Replace content from line_start to line_end with new_string"""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist"
    
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file"    
    
    if line_start < 1 or line_end < line_start:
        return "Error: line_start must be >= 1 and line_end must be >= line_start"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Adjust to 0-based indexing
        start_idx = line_start - 1
        end_idx = line_end - 1
        
        # Check if line numbers are within bounds
        if start_idx >= len(lines) or end_idx >= len(lines):
            return f"Error: Line numbers out of range. File has {len(lines)} lines."
        
        # Split new_string into lines if it contains multiple lines
        new_lines = new_string.splitlines(keepends=True)
        if not new_string.endswith('\n') and new_lines:
            new_lines[-1] += '\n'
        
        # Replace the lines
        updated_lines = lines[:start_idx] + new_lines + lines[end_idx+1:]
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        return f"Successfully replaced lines {line_start}-{line_end} in '{file_path}'"
    except Exception as e:
        return f"Error replacing content in file '{file_path}': {str(e)}"


# Create FastMCP server
mcp = FastMCP("File Tools Optimized Server")

@mcp.tool
def read_file(file_path: str) -> str:
    """Read file content and return with line numbers starting from index 1."""
    return read_file_with_line_numbers(file_path)

@mcp.tool
def replace_str(file_path: str, line_start: int, line_end: int, new_string: str) -> str:
    """Replace content from line_start to line_end with new_string."""
    return replace_str_in_file(file_path, line_start, line_end, new_string)

tool_functions = {
    "read_file": read_file,
    "replace_str": replace_str,
}

if __name__ == "__main__":
    mcp.run()