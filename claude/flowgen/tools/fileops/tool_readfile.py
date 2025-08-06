def read_file(file_path, offset=None, limit=None):
    """
    Read file content with optional line offset and limit.
    
    Args:
        file_path (str): Absolute path to the file to read
        offset (int, optional): Line number to start reading from (1-based)
        limit (int, optional): Maximum number of lines to read
        
    Returns:
        str: File content with line numbers in format "line_number\tcontent"
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
        ValueError: If offset or limit are invalid
    """
    if not file_path.startswith('/'):
        # raise ValueError("file_path must be absolute path")
        return "ValueError: file_path must be absolute path"
    
    if offset is not None and offset < 1:
        # raise ValueError("offset must be >= 1")
        return "ValueError: offset must be >= 1"
    
    if limit is not None and limit < 1:
        # raise ValueError("limit must be >= 1")
        return "ValueError: limit must be >= 1"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # raise FileNotFoundError(f"File not found: {file_path}")
        return f"FileNotFoundError: File not found: {file_path}"
    except PermissionError:
        # raise PermissionError(f"Permission denied: {file_path}")
        return f"PermissionError: Permission denied: {file_path}"
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception:
            # raise ValueError(f"Unable to decode file: {file_path}")
            return f"ValueError: Unable to decode file: {file_path}"
    
    if not lines:
        return "[System Reminder: File exists but has empty contents]"
    
    start_line = (offset - 1) if offset else 0
    end_line = start_line + limit if limit else len(lines)
    
    result = []
    for i, line in enumerate(lines[start_line:end_line], start=start_line + 1):
        line = line.rstrip('\n\r')
        if len(line) > 2000:
            line = line[:1997] + "..."
        result.append(f"{i:5d}\t{line}")
    
    return '\n'.join(result)