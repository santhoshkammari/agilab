import os
import fnmatch
from datetime import datetime
from typing import List, Optional, Dict, Any


def list_directory(path: str, ignore: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    List files and directories in a specified path with optional glob-based filtering.
    
    Args:
        path (str): Absolute path to the directory to list
        ignore (List[str], optional): List of glob patterns to exclude from listing
        
    Returns:
        Dict[str, Any]: Dictionary containing 'entries' list with file/directory information
        
    Raises:
        ValueError: If path is not absolute
        FileNotFoundError: If directory doesn't exist
        PermissionError: If directory can't be accessed
        NotADirectoryError: If path points to a file, not directory
    """
    if not path.startswith('/'):
        # raise ValueError("path must be absolute path")
        return "path must be absolute path"

    if not os.path.exists(path):
        # raise FileNotFoundError(f"Path not found: {path}")
        return f"FileNotFound at Path: {path}"

    if not os.path.isdir(path):
        # raise NotADirectoryError(f"Not a directory: {path}")
        return f"Not a directory: {path}"

    try:
        items = os.listdir(path)
    except PermissionError:
        # raise PermissionError(f"Permission denied: {path}")
        return f"PermissionError: Permission denied: {path}"
        return f"Permission denied: {path}"

    ignore_patterns = ignore or []
    entries = []
    
    for item in items:
        # Check if item matches any ignore pattern
        should_ignore = False
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(item, pattern):
                should_ignore = True
                break
        
        if should_ignore:
            continue
        
        item_path = os.path.join(path, item)
        
        try:
            stat_info = os.stat(item_path)
            is_dir = os.path.isdir(item_path)
            
            entry = {
                "name": item,
                "type": "directory" if is_dir else "file",
                "size": None if is_dir else stat_info.st_size,
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat() + "Z"
            }
            entries.append(entry)
        except (OSError, PermissionError):
            # Skip items we can't stat (e.g., broken symlinks, permission issues)
            continue
    
    # Sort entries: directories first, then files, both alphabetically
    entries.sort(key=lambda x: (x["type"] == "file", x["name"].lower()))
    
    return {"entries": entries}