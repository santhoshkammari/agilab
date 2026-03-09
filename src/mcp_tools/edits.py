from fastmcp import FastMCP
import os


mcp = FastMCP("File Tools Optimized Server")

@mcp.tool
def edit(path: str, s: int, e: int, new: str) -> str:
    """
    Replace a line range in a file.

    Args:
        path: File path
        s: Start line (1-indexed)
        e: End line (1-indexed, inclusive)
        new: Replacement content
    """
    if not os.path.exists(path):
        return f"Error: File not found: '{path}'"
    if not os.path.isfile(path):
        return f"Error: Not a file: '{path}'"

    if s < 1 or e < s:
        return f"Error: s must be >= 1 and e must be >= s"

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        line_count = len(lines)

        if s > line_count or e > line_count:
            return f"Error: Out of range (file has {line_count} lines) — {s}-{e}"

        si, ei = s - 1, e - 1
        result = list(lines[:si])
        if new:
            if not new.endswith('\n'):
                new += '\n'
            result.extend(new.splitlines(keepends=True))
        result.extend(lines[ei + 1:])

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(result)

        return f"Applied edit to '{path}': replaced lines {s}-{e}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
