from .tool_bash import bash_execute
from .tool_edit import edit_file
from .tool_glob import glob_find_files
from .tool_grep import grep_search
from .tool_ls import list_directory
from .tool_multiedit import multi_edit_file
from .tool_readfile import read_file
from .tool_task import task_agent
from .tool_todowrite import todo_write
from .tool_write import write_file

from .tool_bash import bash_execute

tool_functions = {
    "bash_execute": bash_execute,
    "edit_file": edit_file,
    "glob_find_files": glob_find_files,
    "grep_search": grep_search,
    "list_directory": list_directory,
    "multi_edit_file": multi_edit_file,
    "read_file": read_file,
    "task_agent": task_agent,
    "todo_write": todo_write,
    "write_file": write_file,
}