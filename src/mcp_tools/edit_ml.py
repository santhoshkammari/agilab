from fastmcp import FastMCP
import os
from typing import List, Dict, Any


mcp = FastMCP("File Tools Optimized Server")

@mcp.tool
def edit(path: str, edits: List[Dict[str, Any]]) -> str:
    """
    Replace line ranges in a file.

    Args:
        path: File path
        edits: List of edits, each with:
            - s (int): Start line (1-indexed)
            - e (int): End line (1-indexed, inclusive)
            - new (str): Replacement content
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

    # Sort by start line
    sorted_edits = sorted(edits, key=lambda x: x['s'])

    # Check for overlaps
    overlaps = []
    for i in range(len(sorted_edits) - 1):
        a, b = sorted_edits[i], sorted_edits[i + 1]
        if b['s'] <= a['e']:
            overlaps.append(f"{a['s']}-{a['e']} vs {b['s']}-{b['e']}")
    if overlaps:
        return f"Error: Overlapping edits — " + ", ".join(overlaps)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        line_count = len(lines)

        out_of_range = [f"{ed['s']}-{ed['e']}" for ed in sorted_edits if ed['s'] > line_count or ed['e'] > line_count]
        if out_of_range:
            return f"Error: Out of range edits (file has {line_count} lines) — " + ", ".join(out_of_range)

        # Reconstruct file from parts
        result = []
        cursor = 0  # 0-indexed current position

        for ed in sorted_edits:
            si, ei = ed['s'] - 1, ed['e'] - 1
            result.extend(lines[cursor:si])  # unchanged chunk before this edit
            new = ed['new']
            if new:
                if not new.endswith('\n'):
                    new += '\n'
                result.extend(new.splitlines(keepends=True))
            cursor = ei + 1

        result.extend(lines[cursor:])  # remaining lines after last edit

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(result)

        labels = ", ".join(f"{ed['s']}-{ed['e']}" for ed in sorted_edits)
        return f"Applied {len(edits)} edit(s) to '{path}': replaced lines {labels}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
