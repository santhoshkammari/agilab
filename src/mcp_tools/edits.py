from fastmcp import FastMCP
import os
from typing import Optional, Literal

CLIPBOARD = os.path.expanduser("~/.claude_clipboard")

mcp = FastMCP("Edit Server")


def _read(path: str):
    if not os.path.exists(path):
        return None, f"File not found: '{path}'"
    if not os.path.isfile(path):
        return None, f"Not a file: '{path}'"
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines(), None


def _write(path: str, lines: list):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@mcp.tool
def edit(path: str, s: int, e: int, new: str = "") -> str:
    """
    Replace lines s-e in a file with new content. To delete lines, pass new="".

    Args:
        path: File path
        s: Start line (1-indexed)
        e: End line (1-indexed, inclusive)
        new: Replacement content (empty string = delete lines)

    Examples:
        Replace lines 10-12:  edit("app.py", 10, 12, "x = 1\ny = 2")
        Delete lines 5-7:     edit("app.py", 5, 7, "")
        Insert before line 5: edit("app.py", 5, 4, "new line here")
    """
    try:
        lines, err = _read(path)
        if err:
            return f"Error: {err}"
        n = len(lines)
        if s < 1:
            return "Error: s must be >= 1"
        if e < s - 1:
            return "Error: e must be >= s-1 (use e=s-1 for pure insert)"

        si = s - 1
        ei = min(e, n)

        new_lines = []
        if new:
            new_lines = new.splitlines(keepends=True)
            if not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

        result = lines[:si] + new_lines + lines[ei:]
        _write(path, result)
        action = "inserted before" if e < s else f"replaced lines {s}-{e}"
        return f"OK: {action} in '{path}'"
    except Exception as ex:
        return f"Error: {ex}"


@mcp.tool
def yank(path: str, s: int, e: int, mode: Literal["copy", "cut"] = "copy") -> str:
    """
    Yank (copy or cut) lines from a file into the clipboard (~/.claude_clipboard).
    copy = keep lines in file. cut = remove lines from file.

    Args:
        path: File path
        s: Start line (1-indexed)
        e: End line (1-indexed, inclusive)
        mode: "copy" keeps lines intact, "cut" removes them

    Examples:
        Copy lines 5-10:  yank("app.py", 5, 10)
        Cut lines 5-10:   yank("app.py", 5, 10, "cut")
    """
    try:
        lines, err = _read(path)
        if err:
            return f"Error: {err}"
        n = len(lines)
        if s < 1 or e < s or s > n:
            return f"Error: Invalid range {s}-{e} (file has {n} lines)"

        si, ei = s - 1, min(e, n)
        yanked = lines[si:ei]

        with open(CLIPBOARD, "w", encoding="utf-8") as f:
            f.writelines(yanked)

        if mode == "cut":
            _write(path, lines[:si] + lines[ei:])
            return f"Cut {len(yanked)} line(s) from '{path}' lines {s}-{e} → clipboard"
        else:
            return (
                f"Copied {len(yanked)} line(s) from '{path}' lines {s}-{e} → clipboard"
            )
    except Exception as ex:
        return f"Error: {ex}"


@mcp.tool
def put(path: str, at: int, mode: Literal["insert", "replace"] = "insert") -> str:
    """
    Paste clipboard content (~/.claude_clipboard) into a file.
    insert = push existing lines down. replace = overwrite lines starting at 'at'.

    Args:
        path: Target file path
        at: Line number to paste at (1-indexed)
        mode: "insert" adds without removing, "replace" overwrites lines

    Examples:
        Insert clipboard at line 20:  put("app.py", 20)
        Replace from line 20:         put("app.py", 20, "replace")
        Paste into a different file:  put("other.py", 1)
    """
    try:
        if not os.path.exists(CLIPBOARD):
            return "Error: Clipboard is empty. Use yank() first."

        with open(CLIPBOARD, "r", encoding="utf-8") as f:
            clip = f.readlines()

        if not clip:
            return "Error: Clipboard is empty."

        lines, err = _read(path)
        if err:
            return f"Error: {err}"
        n = len(lines)
        at_idx = max(0, min(at - 1, n))

        if mode == "insert":
            result = lines[:at_idx] + clip + lines[at_idx:]
        else:  # replace
            result = lines[:at_idx] + clip + lines[at_idx + len(clip) :]

        _write(path, result)
        return f"Pasted {len(clip)} line(s) into '{path}' at line {at} ({mode})"
    except Exception as ex:
        return f"Error: {ex}"


tool_functions = {
    "edit": edit,
    "yank": yank,
    "put": put,
}

if __name__ == "__main__":
    mcp.run()
