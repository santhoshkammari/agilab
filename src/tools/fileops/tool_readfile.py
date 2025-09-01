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
        return "I need an absolute path to read the file. The path should start with '/' (like /home/user/file.txt)."
    
    if offset is not None and offset < 1:
        # raise ValueError("offset must be >= 1")
        return "The offset parameter should be 1 or greater to specify which line to start reading from."
    
    if limit is not None and limit < 1:
        # raise ValueError("limit must be >= 1")
        return "The limit parameter should be 1 or greater to specify how many lines to read."
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # raise FileNotFoundError(f"File not found: {file_path}")
        return f"I couldn't find the file at {file_path}. Please check if the path is correct and the file exists."
    except PermissionError:
        # raise PermissionError(f"Permission denied: {file_path}")
        return f"I don't have permission to read {file_path}. Please check the file permissions."
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception:
            # raise ValueError(f"Unable to decode file: {file_path}")
            return f"I couldn't read {file_path} because it contains characters that can't be decoded. The file might be binary or use an unsupported encoding."
    
    if not lines:
        return "The file exists but is empty (contains no content)."
    
    start_line = (offset - 1) if offset else 0
    end_line = start_line + limit if limit else len(lines)
    
    result = []
    for i, line in enumerate(lines[start_line:end_line], start=start_line + 1):
        line = line.rstrip('\n\r')
        if len(line) > 2000:
            line = line[:1997] + "..."
        result.append(f"{i:5d}\t{line}")
    
    return '\n'.join(result)