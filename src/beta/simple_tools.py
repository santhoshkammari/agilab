
def read_file(path: str) -> str:
    """Reads the contents of a file and returns it as a string.

    Args:
        path: The file path to read from.

    Returns:
        str: The file contents as a string.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except IOError as e:
        return f"Error reading file {path}: {e}"


def write_file(path: str, content: str) -> str:
    """Writes content to a file.

    Args:
        path: The file path to write to.
        content: The content to write to the file.

    Returns:
        str: A success or error message.
    """
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {path}"
    except IOError as e:
        return f"Error writing to file {path}: {e}"


tools = {
    "read_file": read_file,
    "write_file": write_file
}
