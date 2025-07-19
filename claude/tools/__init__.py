from .__files_advanced import ClaudeTools
from .__web import WebSearchTool

_claude_tools = ClaudeTools()
_web_tools = None  # Will be initialized with browser

def initialize_tools(browser=None):
    """Initialize tools with optional browser for web search"""
    global _web_tools
    if browser and _web_tools is None:
        _web_tools = WebSearchTool(browser)

def get_tools_dict():
    """Get the tools dictionary"""
    return {
        'read_multiple_files': _claude_tools.multi_read.read_multiple_files,
        'read_file': _claude_tools.read.read_file,
        'write_file': _claude_tools.write.write_file,
        'edit_file': _claude_tools.edit.edit_file,
        'apply_edit': _claude_tools.edit.apply_pending_edit,
        'discard_edit': _claude_tools.edit.discard_pending_edit,
        'bash_execute': _claude_tools.bash.bash_execute,
        'glob_find_files': _claude_tools.glob.find_files,
        'grep_search': _claude_tools.grep.grep_search,
        'list_directory': _claude_tools.ls.list_directory,
        'fetch_url': _claude_tools.web_fetch.fetch_url,
        'web_search': _web_tools.web_search if _web_tools else None,
        'todo_read': _claude_tools.todo_read.todo_read,
        'todo_write': _claude_tools.todo_write.todo_write,
        # 'multi_edit_file': _claude_tools.multiedit.multi_edit_file,
        # 'read_notebook': _claude_tools.notebook_read.read_notebook,
        # 'edit_notebook_cell': _claude_tools.notebook_edit.edit_cell,
        # 'task_execute': _claude_tools.task.task_execute
    }

# Backward compatibility
tools_dict = get_tools_dict()