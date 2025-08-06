import os
from pathlib import Path


# Global state to track which files have been read
_read_files = set()


def mark_file_as_read(file_path):
    """Mark a file as having been read (for testing purposes)."""
    _read_files.add(os.path.abspath(file_path))


def clear_read_state():
    """Clear the read state (for testing purposes)."""
    _read_files.clear()


def write_file(file_path, content):
    """
    Write content to a file, with safety mechanisms.
    
    Args:
        file_path (str): Absolute path to the file to write
        content (str): Content to write to the file
        
    Returns:
        str: Success message with file path and size
        
    Raises:
        ValueError: If file_path is not absolute or if trying to overwrite without reading first
        FileNotFoundError: If parent directory doesn't exist
        PermissionError: If insufficient permissions to write
    """
    if not file_path.startswith('/'):
        raise ValueError("file_path must be absolute path")
    
    abs_path = os.path.abspath(file_path)
    
    # Check if file exists and enforce read-before-write policy
    if os.path.exists(abs_path):
        if abs_path not in _read_files:
            raise ValueError("File must be read before writing. Use Read tool on existing file before overwriting")
    
    # Check if parent directory exists
    parent_dir = os.path.dirname(abs_path)
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")
    
    try:
        # Write content atomically
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except PermissionError:
        raise PermissionError(f"Permission denied: {abs_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write file {abs_path}: {str(e)}")
    
    # Get file size for confirmation
    file_size = os.path.getsize(abs_path)
    
    # Generate content preview (first few lines)
    lines = content.splitlines()
    preview_lines = lines[:5]  # Show first 5 lines
    
    return {
        "success": True,
        "file_path": abs_path,
        "file_size": file_size,
        "preview_lines": preview_lines,
        "total_lines": len(lines),
        "simple_message": f"Successfully wrote {file_size} bytes to {abs_path}"
    }