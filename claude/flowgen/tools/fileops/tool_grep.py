import os
import subprocess
import shlex
from typing import Optional, Union, Literal


def grep_search(
    pattern: str,
    path: Optional[str] = None,
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
    glob: Optional[str] = None,
    type: Optional[str] = None,
    i: Optional[bool] = None,
    n: Optional[bool] = None,
    A: Optional[int] = None,
    B: Optional[int] = None,
    C: Optional[int] = None,
    multiline: Optional[bool] = None,
    head_limit: Optional[int] = None
) -> str:
    """
    Powerful search tool built on ripgrep for finding patterns in files.
    
    Args:
        pattern (str): Regular expression pattern to search for
        path (str, optional): File or directory to search in. Defaults to current working directory.
        output_mode (str): Output mode - "content", "files_with_matches", or "count". Defaults to "files_with_matches".
        glob (str, optional): Glob pattern to filter files (e.g., "*.js", "*.{ts,tsx}")
        type (str, optional): File type to search (e.g., "js", "py", "rust", "go")
        i (bool, optional): Case insensitive search
        n (bool, optional): Show line numbers in output (content mode only)
        A (int, optional): Number of lines to show after each match (content mode only)
        B (int, optional): Number of lines to show before each match (content mode only)
        C (int, optional): Number of lines to show before and after each match (content mode only)
        multiline (bool, optional): Enable multiline mode where . matches newlines
        head_limit (int, optional): Limit output to first N entries
        
    Returns:
        str: Search results formatted according to output_mode
        
    Raises:
        ValueError: If pattern is empty or parameters are invalid
        FileNotFoundError: If specified path doesn't exist
        subprocess.CalledProcessError: If ripgrep command fails
    """
    if not pattern:
        # raise ValueError("pattern parameter is required and cannot be empty")
        return "ValueError: pattern parameter is required and cannot be empty"
    
    # Validate path if provided
    if path and not os.path.exists(path):
        # raise FileNotFoundError(f"Path not found: {path}")
        return f"FileNotFoundError: Path not found: {path}"
    
    # Validate context options only work with content mode
    if output_mode != "content":
        if n is not None and n:
            # raise ValueError("-n (line numbers) requires output_mode: 'content'")
            return "ValueError: -n (line numbers) requires output_mode: 'content'"
        if A is not None:
            # raise ValueError("-A (after context) requires output_mode: 'content'")
            return "ValueError: -A (after context) requires output_mode: 'content'"
        if B is not None:
            # raise ValueError("-B (before context) requires output_mode: 'content'")
            return "ValueError: -B (before context) requires output_mode: 'content'"
        if C is not None:
            # raise ValueError("-C (context) requires output_mode: 'content'")
            return "ValueError: -C (context) requires output_mode: 'content'"
    
    # Build ripgrep command
    cmd = ["rg"]
    
    # Add pattern
    cmd.append(pattern)
    
    # Set output mode
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")
    # content mode is default, no flag needed
    
    # Add file filtering
    if glob:
        cmd.extend(["--glob", glob])
    
    if type:
        cmd.extend(["--type", type])
    
    # Add search modifiers
    if i:
        cmd.append("--ignore-case")
    
    if multiline:
        cmd.extend(["--multiline", "--multiline-dotall"])
    
    # Add context options (only for content mode)
    if output_mode == "content":
        if n:
            cmd.append("--line-number")
        if A is not None:
            cmd.extend(["-A", str(A)])
        if B is not None:
            cmd.extend(["-B", str(B)])
        if C is not None:
            cmd.extend(["-C", str(C)])
    
    # Add path if specified
    if path:
        cmd.append(path)
    
    # Execute ripgrep command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit (no matches is exit code 1)
        )
        
        # Handle no matches (exit code 1 is normal for no matches)
        if result.returncode == 1:
            return ""
        
        # Handle other errors
        if result.returncode > 1:
            error_msg = f"Command failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr.strip()}"
            # raise subprocess.CalledProcessError(
            #     result.returncode, 
            #     cmd, 
            #     output=result.stdout, 
            #     stderr=result.stderr
            # )
            return f"subprocess.CalledProcessError: Command failed with exit code {result.returncode}: {result.stderr.strip() if result.stderr else 'Unknown error'}"
        
        output = result.stdout.strip()
        
        # Apply head_limit if specified
        if head_limit is not None and head_limit > 0:
            lines = output.split('\n')
            if lines and lines[0]:  # Only if there's actual content
                limited_lines = lines[:head_limit]
                output = '\n'.join(limited_lines)
        
        return output
        
    except FileNotFoundError:
        # raise FileNotFoundError("ripgrep (rg) command not found. Please install ripgrep.")
        return "FileNotFoundError: ripgrep (rg) command not found. Please install ripgrep."
    except subprocess.CalledProcessError as e:
        error_msg = f"ripgrep command failed: {e.stderr}" if e.stderr else f"ripgrep command failed with exit code {e.returncode}"
        # raise subprocess.CalledProcessError(e.returncode, e.cmd, e.output, e.stderr) from e
        return f"subprocess.CalledProcessError: ripgrep command failed: {e.stderr if e.stderr else f'exit code {e.returncode}'}"


# Alias for backwards compatibility and convenience
def grep(
    pattern: str,
    path: Optional[str] = None,
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
    **kwargs
) -> str:
    """
    Convenience alias for grep_search function.
    """
    return grep_search(pattern, path, output_mode, **kwargs)