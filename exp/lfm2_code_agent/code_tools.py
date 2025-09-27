"""Simple Python functions for LLM code manipulation."""


def read_file(filepath: str) -> str:
    """Read the contents of a file.

    Args:
        filepath: Path to the file to read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath: str, content: str) -> None:
    """Write content to a file.

    Args:
        filepath: Path to the file to write.
        content: Content to write to the file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def delete_lines(filepath: str, start: int, end: int) -> None:
    """Delete lines from a file between start and end (inclusive).

    Args:
        filepath: Path to the file to modify.
        start: Starting line number (1-indexed).
        end: Ending line number (1-indexed, inclusive).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If start or end line numbers are invalid.
        IOError: If there's an error reading or writing the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if start < 1 or end < 1 or start > len(lines) or end > len(lines):
        raise ValueError(f"Invalid line numbers: start={start}, end={end}, file has {len(lines)} lines")

    if start > end:
        raise ValueError(f"Start line ({start}) cannot be greater than end line ({end})")

    # Convert to 0-indexed and delete lines
    del lines[start-1:end]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)


if __name__ == "__main__":
    # Test the functions
    print("Testing code_tools functions:")

    # Test write and read
    test_file = "test_file.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"

    try:
        write_file(test_file, test_content)
        content = read_file(test_file)
        print(f"✓ write_file and read_file: successful")

        # Test delete_lines
        delete_lines(test_file, 2, 3)  # Delete lines 2-3
        updated_content = read_file(test_file)
        print(f"✓ delete_lines: successful")
        print(f"Original lines: {len(test_content.splitlines())}")
        print(f"After deletion: {len(updated_content.splitlines())}")

        # Clean up
        import os
        os.remove(test_file)

    except Exception as e:
        print(f"✗ Error: {e}")