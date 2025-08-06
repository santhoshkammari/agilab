from .tool_bash import bash_execute
from .tool_edit import edit_file
from .tool_glob import glob_find_files
from .tool_grep import grep_search
from .tool_ls import list_directory
from .tool_multiedit import multi_edit_file
from .tool_readfile import read_file, read_multiple_files
from .tool_task import task_agent
from .tool_todowrite import todo_read, todo_write
from .tool_webfetch import fetch_url
from .tool_websearch import async_web_search
from .tool_write import write_file

tool_functions = {
    "bash_execute": bash_execute,
    "edit_file": edit_file,
    "glob_find_files": glob_find_files,
    "grep_search": grep_search,
    "list_directory": list_directory,
    "multi_edit_file": multi_edit_file,
    "read_file": read_file,
    "read_multiple_files": read_multiple_files,
    "task_agent": task_agent,
    "todo_read": todo_read,
    "todo_write": todo_write,
    "fetch_url": fetch_url,
    "async_web_search": async_web_search,
    "write_file": write_file,
}