def multi_edit(file_path, edits):
    """
    Perform multiple find-and-replace operations on a single file in one atomic transaction.
    
    Args:
        file_path (str): Absolute path to the file to modify
        edits (list): List of edit objects, each containing:
            - old_string (str): Exact text to find and replace
            - new_string (str): Replacement text
            - replace_all (bool, optional): Replace all occurrences (default: False)
    
    Returns:
        str: Success message indicating the number of edits applied
        
    Raises:
        ValueError: If file_path is not absolute, if edits is empty, if old_string equals new_string,
                   or if any old_string is not found in the current file content
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read or written
    """
    if not file_path.startswith('/'):
        raise ValueError("file_path must be absolute path")
    
    if not edits:
        raise ValueError("edits list cannot be empty")
    
    # Validate all edits before processing
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise ValueError(f"Edit {i} must be a dictionary")
        
        if 'old_string' not in edit or 'new_string' not in edit:
            raise ValueError(f"Edit {i} must contain 'old_string' and 'new_string'")
        
        old_string = edit['old_string']
        new_string = edit['new_string']
        
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise ValueError(f"Edit {i}: old_string and new_string must be strings")
        
        if old_string == new_string:
            raise ValueError(f"Edit {i}: old_string and new_string must be different")
    
    # Read the original file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied reading file: {file_path}")
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception:
            raise ValueError(f"Unable to decode file: {file_path}")
    
    # Store original content for rollback
    original_content = content
    
    # Apply edits sequentially
    try:
        for i, edit in enumerate(edits):
            old_string = edit['old_string']
            new_string = edit['new_string']
            replace_all = edit.get('replace_all', False)
            
            if replace_all:
                if old_string not in content:
                    raise ValueError(f"Edit {i}: String not found: '{old_string}'")
                content = content.replace(old_string, new_string)
            else:
                # Check if string exists and is unique
                occurrences = content.count(old_string)
                if occurrences == 0:
                    raise ValueError(f"Edit {i}: String not found: '{old_string}'")
                elif occurrences > 1:
                    raise ValueError(f"Edit {i}: String '{old_string}' appears {occurrences} times. Use replace_all=True or provide more context for unique match")
                
                content = content.replace(old_string, new_string, 1)
        
        # Write the modified content back to the file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except PermissionError:
            raise PermissionError(f"Permission denied writing to file: {file_path}")
        
        return f"Successfully applied {len(edits)} edits to {file_path}"
        
    except Exception as e:
        # Rollback: restore original content on any failure
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
        except Exception:
            # If rollback fails, at least preserve the error message
            pass
        raise e


def create_file_with_multi_edit(file_path, content):
    """
    Create a new file using the MultiEdit pattern with empty old_string.
    
    Args:
        file_path (str): Absolute path to the new file to create
        content (str): Initial content for the new file
        
    Returns:
        str: Success message
        
    Raises:
        ValueError: If file_path is not absolute or if file already exists
        PermissionError: If file can't be created
    """
    if not file_path.startswith('/'):
        raise ValueError("file_path must be absolute path")
    
    # Check if file already exists
    import os
    if os.path.exists(file_path):
        raise ValueError(f"File already exists: {file_path}")
    
    # Create directory if it doesn't exist
    import os
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except PermissionError:
            raise PermissionError(f"Permission denied creating directory: {directory}")
    
    # Create the file using multi_edit pattern
    try:
        # Create empty file first
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        # Use multi_edit to add content
        edits = [{"old_string": "", "new_string": content}]
        return multi_edit(file_path, edits)
        
    except PermissionError:
        raise PermissionError(f"Permission denied creating file: {file_path}")