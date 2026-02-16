from fastmcp import FastMCP
import os
from typing import Optional, List, Dict, Any


mcp = FastMCP("File Tools Optimized Server")

@mcp.tool
def edit(path: str, edits: List[Dict[str, Any]]) -> str:
    """
    Replace line ranges in a file.
    edits: [{s, e, new}] — applied bottom-to-top, 1-indexed.
    Example: [{"s": 5, "e": 7, "new": "replacement\\n"}, {"s": 10, "e": 10, "new": "line 10\\n"}]
    """
    if not os.path.exists(path):
        return f"Error: File not found: '{path}'"
    if not os.path.isfile(path):
        return f"Error: Not a file: '{path}'"

    for i, ed in enumerate(edits):
        if 's' not in ed or 'e' not in ed or 'new' not in ed:
            return f"Error: Edit {i+1} missing fields (s, e, new)"
        if ed['s'] < 1 or ed['e'] < ed['s']:
            return f"Error: Edit {i+1}: s must be >= 1 and e must be >= s"

    sorted_edits = sorted(edits, key=lambda x: x['s'], reverse=True)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        results = []

        for ed in sorted_edits:
            s, e, new = ed['s'], ed['e'], ed['new']
            si, ei = s - 1, e - 1

            if si >= len(lines) or ei >= len(lines):
                return f"Error: Lines {s}-{e} out of range. File has {len(lines)} lines."

            if new:
                if not new.endswith('\n'):
                    new = new + '\n'
                new_lines = new.splitlines(keepends=True)
            else:
                new_lines = []

            lines = lines[:si] + new_lines + lines[ei+1:]
            results.append(f"Replaced {s}-{e}")

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"Applied {len(edits)} edit(s) to '{path}': " + ", ".join(results)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
