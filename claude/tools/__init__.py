from ._files import FilesystemTools
from .__files_advanced import ClaudeTools
from .__web import WebSearchTool
import asyncio

_fs_tools = FilesystemTools()
_claude_tools = ClaudeTools()
_web_tools = WebSearchTool()

basic_filsystem_tools = [
    _fs_tools.read_file,
    _fs_tools.write_file,
    _fs_tools.list_directory,
    _fs_tools.create_directory,
    _fs_tools.copy_file,
    _fs_tools.move_file,
    _fs_tools.search_files,
    _fs_tools.get_file_info,
    _fs_tools.list_allowed_directories,
    _fs_tools.read_multiple_files,
    _fs_tools.tree,
    _fs_tools.delete_file,
    _fs_tools.modify_file,
    _fs_tools.search_within_files,
    _fs_tools.handle_read_resource
]

# Advanced tools wrapper functions
def read_file(file_path: str, encoding: str = 'utf-8', offset: int = None, limit: int = None):
    """Read file contents with advanced features"""
    return asyncio.run(_claude_tools.read.read_file(file_path, encoding, offset, limit))

def write_file(file_path: str, content: str, encoding: str = 'utf-8', 
                       create_directories: bool = True, overwrite: bool = True):
    """Write content to a file with advanced features"""
    return asyncio.run(_claude_tools.write.write_file(file_path, content, encoding, create_directories, overwrite))

def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Edit file by replacing strings with advanced features"""
    return asyncio.run(_claude_tools.edit.edit_file(file_path, old_string, new_string, replace_all))

def multi_edit_file(file_path: str, edits: list):
    """Apply multiple edits to a file atomically"""
    return asyncio.run(_claude_tools.multiedit.multi_edit_file(file_path, edits))

def bash_execute(command: str, cwd: str = None, env: dict = None):
    """Execute shell commands safely with advanced features"""
    return asyncio.run(_claude_tools.bash.execute(command, cwd, env))

def glob_find_files(pattern: str, path: str = ".", recursive: bool = True):
    """Find files matching a glob pattern with advanced features"""
    return asyncio.run(_claude_tools.glob.find_files(pattern, path, recursive))

def grep_search(pattern: str, path: str = ".", file_pattern: str = "*", 
                        case_sensitive: bool = False, max_results: int = 1000):
    """Search for pattern in files with advanced features"""
    return asyncio.run(_claude_tools.grep.search(pattern, path, file_pattern, case_sensitive, max_results))

def list_directory(path: str = ".", show_hidden: bool = False, 
                           recursive: bool = False, ignore_patterns: list = None):
    """List directory contents with advanced features"""
    return asyncio.run(_claude_tools.ls.list_directory(path, show_hidden, recursive, ignore_patterns))

def read_notebook(notebook_path: str):
    """Read Jupyter notebook files with advanced features"""
    return asyncio.run(_claude_tools.notebook_read.read_notebook(notebook_path))

def edit_notebook_cell(notebook_path: str, cell_number: int, new_source: str, 
                                cell_type: str = 'code', edit_mode: str = 'replace'):
    """Edit a specific cell in a Jupyter notebook with advanced features"""
    return asyncio.run(_claude_tools.notebook_edit.edit_cell(notebook_path, cell_number, new_source, cell_type, edit_mode))

def web_fetch(url: str, prompt: str = None):
    """Fetch content from URL with advanced features"""
    return asyncio.run(_claude_tools.web_fetch.fetch_url(url, prompt))

def web_search(query: str, max_results: int = 10):
    """Search the web with advanced features"""
    return asyncio.run(_web_tools.search(query, max_results))

def todo_read():
    """Read current todo list with advanced features"""
    return asyncio.run(_claude_tools.todo_read.read_todos())

def todo_write(todos: list):
    """Write/update the todo list with advanced features, max 10 todos each with max of 5 to 6 words each"""
    return asyncio.run(_claude_tools.todo_write.write_todos(todos))

def task_execute(description: str, instructions: str):
    """Execute a complex task with multiple steps"""
    return asyncio.run(_claude_tools.task.execute_task(description, instructions))

# Advanced tools list for LLM integration
tools = [
    read_file,
    write_file,
    edit_file,
    multi_edit_file,
    bash_execute,
    glob_find_files,
    grep_search,
    list_directory,
    read_notebook,
    edit_notebook_cell,
    web_fetch,
    web_search,
    todo_read,
    todo_write,
    task_execute
]