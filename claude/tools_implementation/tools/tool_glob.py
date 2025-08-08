import os
import glob
from pathlib import Path
from typing import List, Optional


def glob_files(pattern: str, path: Optional[str] = None) -> List[str]:
    """
    Find files matching a glob pattern, sorted by modification time.
    
    Args:
        pattern (str): Glob pattern to match files against
        path (str, optional): Directory to search in. Defaults to current working directory.
        
    Returns:
        List[str]: List of absolute file paths matching the pattern, sorted by modification time (newest first)
        
    Raises:
        ValueError: If pattern is empty or path is invalid
        PermissionError: If directory can't be accessed
        OSError: If filesystem error occurs
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty")
    
    # Use current working directory if path not provided
    search_path = path if path is not None else os.getcwd()
    
    # Validate path exists and is a directory
    if not os.path.exists(search_path):
        raise ValueError(f"Path does not exist: {search_path}")
    
    if not os.path.isdir(search_path):
        raise ValueError(f"Path is not a directory: {search_path}")
    
    try:
        # Change to the search directory for glob operation
        original_cwd = os.getcwd()
        os.chdir(search_path)
        
        try:
            # Use glob with recursive=True to support ** patterns
            matches = glob.glob(pattern, recursive=True)
            
            # Filter out directories, keep only files
            file_matches = [match for match in matches if os.path.isfile(match)]
            
            # Convert to absolute paths
            absolute_paths = [os.path.abspath(match) for match in file_matches]
            
            # Sort by modification time (newest first)
            def get_mtime(filepath):
                try:
                    return os.path.getmtime(filepath)
                except (OSError, FileNotFoundError):
                    # If we can't get mtime, put at end
                    return 0
            
            sorted_paths = sorted(absolute_paths, key=get_mtime, reverse=True)
            
            return sorted_paths
            
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            
    except PermissionError as e:
        raise PermissionError(f"Permission denied accessing directory: {search_path}")
    except OSError as e:
        raise OSError(f"Filesystem error: {e}")


def glob_files_advanced(pattern: str, path: Optional[str] = None, 
                       include_dirs: bool = False, max_results: Optional[int] = None) -> List[str]:
    """
    Advanced glob function with additional options.
    
    Args:
        pattern (str): Glob pattern to match files against
        path (str, optional): Directory to search in. Defaults to current working directory.
        include_dirs (bool): Whether to include directories in results. Defaults to False.
        max_results (int, optional): Maximum number of results to return
        
    Returns:
        List[str]: List of absolute paths matching the pattern, sorted by modification time (newest first)
        
    Raises:
        ValueError: If pattern is empty or path is invalid
        PermissionError: If directory can't be accessed
        OSError: If filesystem error occurs
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty")
    
    # Use current working directory if path not provided
    search_path = path if path is not None else os.getcwd()
    
    # Validate path exists and is a directory
    if not os.path.exists(search_path):
        raise ValueError(f"Path does not exist: {search_path}")
    
    if not os.path.isdir(search_path):
        raise ValueError(f"Path is not a directory: {search_path}")
    
    try:
        # Change to the search directory for glob operation
        original_cwd = os.getcwd()
        os.chdir(search_path)
        
        try:
            # Use glob with recursive=True to support ** patterns
            matches = glob.glob(pattern, recursive=True)
            
            # Filter based on include_dirs setting
            if include_dirs:
                filtered_matches = matches  # Include everything
            else:
                filtered_matches = [match for match in matches if os.path.isfile(match)]
            
            # Convert to absolute paths
            absolute_paths = [os.path.abspath(match) for match in filtered_matches]
            
            # Sort by modification time (newest first), then alphabetically
            def sort_key(filepath):
                try:
                    mtime = os.path.getmtime(filepath)
                    return (-mtime, filepath)  # Negative mtime for descending order
                except (OSError, FileNotFoundError):
                    # If we can't get mtime, sort by name only at end
                    return (0, filepath)
            
            sorted_paths = sorted(absolute_paths, key=sort_key)
            
            # Apply max_results limit if specified
            if max_results is not None and max_results > 0:
                sorted_paths = sorted_paths[:max_results]
            
            return sorted_paths
            
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            
    except PermissionError as e:
        raise PermissionError(f"Permission denied accessing directory: {search_path}")
    except OSError as e:
        raise OSError(f"Filesystem error: {e}")