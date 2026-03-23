"""
Built-in tools for the agent loop — same names/descriptions as Claude Code tools.
Each tool is a dict with: name, description, input_schema, handler (sync callable).
"""

import os
import subprocess
import glob as glob_module
import re


def _text(content: str) -> dict:
    return {"type": "text", "text": content}


def _result(text: str) -> dict:
    return {"content": [_text(text)]}


# ── handlers ──────────────────────────────────────────────────────────────────

def _read(file_path: str, offset: int = 0, limit: int = 2000) -> dict:
    try:
        with open(file_path) as f:
            lines = f.readlines()
        chunk = lines[offset: offset + limit]
        numbered = "".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(chunk))
        return _result(numbered or "(empty)")
    except Exception as e:
        return _result(f"Error: {e}")


def _write(file_path: str, content: str) -> dict:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return _result(f"Written {file_path}")
    except Exception as e:
        return _result(f"Error: {e}")


def _edit(file_path: str, old_string: str, new_string: str) -> dict:
    try:
        with open(file_path) as f:
            text = f.read()
        if old_string not in text:
            return _result(f"Error: old_string not found in {file_path}")
        new_text = text.replace(old_string, new_string, 1)
        with open(file_path, "w") as f:
            f.write(new_text)
        
        # Get git diff output to show changes
        diff_result = subprocess.run(
            ["git", "diff", "--no-color", file_path],
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if diff_result.stdout:
            colored_lines = []
            for line in diff_result.stdout.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    colored_lines.append(f"\033[1m{line}\033[0m")
                elif line.startswith("+"):
                    colored_lines.append(f"\033[32m{line}\033[0m")
                elif line.startswith("-"):
                    colored_lines.append(f"\033[31m{line}\033[0m")
                elif line.startswith("@@"):
                    colored_lines.append(f"\033[36m{line}\033[0m")
                else:
                    colored_lines.append(line)
            diff_output = "\n".join(colored_lines)
        else:
            diff_output = "(no changes to track)"

        return _result(f"Edited {file_path}\n\n{diff_output}")
    except Exception as e:
        return _result(f"Error: {e}")


def _bash(command: str, timeout: int = 30) -> dict:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
        return _result(out or "(no output)")
    except subprocess.TimeoutExpired:
        return _result("Error: command timed out")
    except Exception as e:
        return _result(f"Error: {e}")


def _glob(pattern: str, path: str = ".") -> dict:
    matches = glob_module.glob(os.path.join(path, pattern), recursive=True)
    return _result("\n".join(sorted(matches)) or "(no matches)")


def _grep(pattern: str, path: str = ".", glob: str = "*") -> dict:
    try:
        r = subprocess.run(
            ["grep", "-rn", "--include", glob, pattern, path],
            capture_output=True, text=True, timeout=15,
        )
        return _result(r.stdout or "(no matches)")
    except Exception as e:
        return _result(f"Error: {e}")


def _multiedit(file_path: str, replacements: list) -> dict:
    try:
        if isinstance(replacements, str):
            # Handle backward compatible single replacement
            replacements = [{"old_string": replacements}]
        
        # Validate replacements
        for i, r in enumerate(replacements):
            if "old_string" not in r:
                return _result(f"Error: replacement #{i+1} missing 'old_string'")
            if not isinstance(r.get("old_string"), str):
                return _result(f"Error: old_string must be a string")
        
        with open(file_path) as f:
            text = f.read()
        
        modified = False
        for r in replacements:
            old_string = r["old_string"]
            new_string = r.get("new_string", "")
            
            if old_string not in text:
                return _result(f"Error: {old_string!r} not found in {file_path}")
            
            text = text.replace(old_string, new_string, 1)
            modified = True
        
        if not modified:
            return _result("No modifications made (old_string not found)")
        
        with open(file_path, "w") as f:
            f.write(text)
        
        # Get git diff output to show changes
        diff_result = subprocess.run(
            ["git", "diff", "--no-color", file_path],
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if diff_result.stdout:
            colored_lines = []
            for line in diff_result.stdout.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    colored_lines.append(f"\033[1m{line}\033[0m")
                elif line.startswith("+"):
                    colored_lines.append(f"\033[32m{line}\033[0m")
                elif line.startswith("-"):
                    colored_lines.append(f"\033[31m{line}\033[0m")
                elif line.startswith("@@"):
                    colored_lines.append(f"\033[36m{line}\033[0m")
                else:
                    colored_lines.append(line)
            diff_output = "\n".join(colored_lines)
        else:
            diff_output = "(no changes to track)"

        return _result(f"Edited {file_path}\n\n{diff_output}")
    except Exception as e:
        return _result(f"Error: {e}")


# ── tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from the local filesystem. Returns file contents with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line number to start reading from (0-indexed)", "default": 0},
                "limit": {"type": "integer", "description": "Max number of lines to read", "default": 2000},
            },
            "required": ["file_path"],
        },
        "handler": lambda args: _read(args["file_path"], args.get("offset", 0), args.get("limit", 2000)),
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating it or overwriting it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
        "handler": lambda args: _write(args["file_path"], args["content"]),
    },
    {
        "name": "Edit",
        "description": "Exact string replacement in a file. Replaces first occurrence of old_string with new_string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        "handler": lambda args: _edit(args["file_path"], args["old_string"], args["new_string"]),
    },
    {
        "name": "Bash",
        "description": "Execute a bash command and return stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["command"],
        },
        "handler": lambda args: _bash(args["command"], args.get("timeout", 30)),
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
        },
        "handler": lambda args: _glob(args["pattern"], args.get("path", ".")),
    },
    {
        "name": "Grep",
        "description": "Search file contents using a regex pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "default": "*"},
            },
            "required": ["pattern"],
        },
        "handler": lambda args: _grep(args["pattern"], args.get("path", "."), args.get("glob", "*")),
    },
    {
        "name": "Multiedit",
        "description": "Perform multiple string replacements in a file at once. Replaces the first occurrence of each old_string with corresponding new_string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string", "description": "The string to find and replace"},
                            "new_string": {"type": "string", "description": "The replacement string"}
                        },
                        "required": ["old_string"]
                    },
                    "description": "List of replacements to perform. Each object must have 'old_string', optionally with 'new_string'"
                }
            },
            "required": ["file_path", "replacements"],
        },
        "handler": lambda args: _multiedit(args["file_path"], args["replacements"]),
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}

# OpenAI-compatible schema for vLLM /v1/chat/completions
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]
