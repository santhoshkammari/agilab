import os
from typing import Optional
import difflib


# Global state to track read files for safety enforcement
_read_files = set()


def mark_file_as_read(file_path: str):
    """
    Mark a file as read for read-before-edit policy enforcement.
    
    Args:
        file_path (str): Absolute path to the file that was read
    """
    if file_path.startswith('/'):
        _read_files.add(file_path)


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False):
    """
    Perform exact string replacement in a file with safety mechanisms.
    
    Args:
        file_path (str): Absolute path to the file to modify
        old_string (str): Exact text to find and replace
        new_string (str): Replacement text
        replace_all (bool): Replace all occurrences instead of requiring uniqueness
        
    Returns:
        str: Success message with details about the replacement
        
    Raises:
        ValueError: If file_path is not absolute, strings are identical, or validation fails
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be written
        RuntimeError: If file wasn't read first, string not found, or multiple matches without replace_all
    """
    # Validate file path is absolute
    if not file_path.startswith('/'):
        # raise ValueError("file_path must be absolute path")
        return "I need an absolute path to edit the file. The path should start with '/' (like /home/user/file.txt)."
    
    # Enforce read-before-edit policy
    if file_path not in _read_files:
        # raise RuntimeError("File must be read before editing. Use Read tool on file before attempting edits.")
        return "I need to read the file first before I can edit it. Please use the Read tool on this file before attempting edits."
    
    # Validate strings are different
    if old_string == new_string:
        # raise ValueError("old_string and new_string are identical. No changes would be made.")
        return "The old text and new text are identical, so no changes would be made. Please provide different text for the replacement."
    
    # Check file exists and is readable
    if not os.path.exists(file_path):
        # raise FileNotFoundError(f"File not found: {file_path}")
        return f"I couldn't find the file at {file_path}. Please check if the path is correct and the file exists."
    
    if not os.access(file_path, os.R_OK):
        # raise PermissionError(f"File not readable: {file_path}")
        return f"I don't have permission to read {file_path}. Please check the file permissions."
    
    if not os.access(file_path, os.W_OK):
        # raise PermissionError(f"File not writable: {file_path}")
        return f"I don't have permission to write to {file_path}. Please check the file permissions."
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception:
            # raise ValueError(f"Unable to decode file: {file_path}")
            return f"I couldn't read {file_path} because it contains characters that can't be decoded. The file might be binary or use an unsupported encoding."
    
    # Count occurrences of old_string
    occurrence_count = content.count(old_string)
    
    if occurrence_count == 0:
        # raise RuntimeError(f"String not found in file: '{old_string[:100]}{'...' if len(old_string) > 100 else ''}'")
        return f"I couldn't find the text '{old_string[:100]}{'...' if len(old_string) > 100 else ''}' in the file. Please check that the text matches exactly, including whitespace and formatting."
    
    if occurrence_count > 1 and not replace_all:
        # raise RuntimeError(f"Multiple matches found ({occurrence_count} occurrences). Use replace_all=True to replace all instances or provide more context to make the string unique.")
        return f"I found {occurrence_count} matches for that text. To replace all instances, set replace_all=True, or provide more context to make the text unique."
    
    # Perform replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replaced_count = occurrence_count
    else:
        # Single replacement (we know there's exactly one occurrence)
        new_content = content.replace(old_string, new_string, 1)
        replaced_count = 1
    
    # IMPROVED DIFF GENERATION
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    # Generate unified diff with context lines
    diff_lines = list(difflib.unified_diff(
        old_lines, 
        new_lines, 
        lineterm='',
        n=3  # Show 3 lines of context
    ))
    
    # Process diff lines to extract meaningful information with better formatting
    meaningful_diff = []
    line_info = []
    
    for line in diff_lines:
        if line.startswith('@@'):
            # Extract line number information
            line_info.append(line.strip())
            meaningful_diff.append(line.strip())
        elif line.startswith('-') and not line.startswith('---'):
            meaningful_diff.append(f"- {line[1:].rstrip()}")
        elif line.startswith('+') and not line.startswith('+++'):
            meaningful_diff.append(f"+ {line[1:].rstrip()}")
        elif line.startswith(' '):  # Context lines
            meaningful_diff.append(f"  {line[1:].rstrip()}")
    
    # Write updated content back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except PermissionError:
        # raise PermissionError(f"Permission denied writing to file: {file_path}")
        return f"I don't have permission to write to {file_path}. Please check the file permissions."
    except Exception as e:
        # raise RuntimeError(f"Error writing to file {file_path}: {str(e)}")
        return f"I encountered an error while writing to {file_path}: {str(e)}"
    
    # Return structured data for rich display
    return {
        "success": True,
        "file_path": file_path,
        "changes_count": replaced_count,
        "diff_lines": meaningful_diff,  # No limit, show all diff lines
        "line_info": line_info,  # Include line number information
        "has_changes": len([l for l in meaningful_diff if l.startswith(('+', '-'))]) > 0,
        "simple_message": f"{'Replaced' if replaced_count == 1 else f'Replaced all {replaced_count} occurrences of'} text in {file_path}. Changes applied successfully."
    }


def clear_read_files():
    """
    Clear the read files tracking for testing purposes.
    """
    global _read_files
    _read_files.clear()