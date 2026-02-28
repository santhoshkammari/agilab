from langchain_community.tools.file_management import (
    ReadFileTool, WriteFileTool, ListDirectoryTool,
    CopyFileTool, MoveFileTool, DeleteFileTool
)

_read = ReadFileTool()
_write = WriteFileTool()
_list = ListDirectoryTool()
_copy = CopyFileTool()
_move = MoveFileTool()
_delete = DeleteFileTool()


def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read.
    """
    return _read.run(file_path)


def write_file(file_path: str, text: str) -> str:
    """Write text to a file, creating it if it doesn't exist.

    Args:
        file_path: Path to the file to write.
        text: Text content to write to the file.
    """
    return _write.run({"file_path": file_path, "text": text})


def list_directory(dir_path: str = ".") -> str:
    """List files and directories inside a directory.

    Args:
        dir_path: Path to the directory to list.
    """
    return _list.run(dir_path)


def copy_file(source_path: str, destination_path: str) -> str:
    """Copy a file from source to destination.

    Args:
        source_path: Path of the file to copy.
        destination_path: Path to save the copied file.
    """
    return _copy.run({"source_path": source_path, "destination_path": destination_path})


def move_file(source_path: str, destination_path: str) -> str:
    """Move a file from source to destination.

    Args:
        source_path: Path of the file to move.
        destination_path: New path for the moved file.
    """
    return _move.run({"source_path": source_path, "destination_path": destination_path})


def delete_file(file_path: str) -> str:
    """Delete a file at the given path.

    Args:
        file_path: Path of the file to delete.
    """
    return _delete.run(file_path)


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file with new content.

    Args:
        file_path: Path to the file to edit.
        old_string: The exact string to find and replace.
        new_string: The string to replace it with.
    """
    with open(file_path, "r") as f:
        content = f.read()
    if old_string not in content:
        return f"Error: string not found in {file_path}"
    new_content = content.replace(old_string, new_string, 1)
    with open(file_path, "w") as f:
        f.write(new_content)
    return f"Successfully edited {file_path}"


tools = [read_file, write_file, edit_file, list_directory, copy_file, move_file, delete_file]
