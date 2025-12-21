import subprocess
import json
from typing import List, Iterator, Dict, Any, Optional
from fastmcp import FastMCP

# Default directories to include for system access
INCLUDE_DIRECTORIES = ['/home', '/tmp', '/etc', '/var', '/opt']

def qwen_stream(args: List[str]) -> Iterator[Dict[str, Any]]:
    cmd = ['qwen', '-y', '--output-format', 'stream-json']

    # Add default include directories
    for directory in INCLUDE_DIRECTORIES:
        cmd.extend(['--include-directories', directory])

    cmd.extend(args)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    for stdout_line in iter(process.stdout.readline, ''):
        line = stdout_line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        stderr_output = process.stderr.read() if process.stderr else ""
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_output)


mcp = FastMCP("Qwen Agent")

@mcp.tool
def task_subagent(prompt: str, model: Optional[str] = None, sandbox: bool = False):
    """
    Execute a task using an active sub-agent backed by a capable code model.

    This sub-agent is responsible for performing actions and executing code
    in YOLO mode (auto-accepts actions). The underlying model can be swapped
    without changing the interface, allowing any caller model to delegate
    execution to this agent.

    Args:
        prompt: The task or instruction for the sub-agent to execute.
        model: Optional backend model identifier to use for execution.
        sandbox: Whether to run the sub-agent in sandbox (restricted) mode.

    Returns:
        The final result produced by the sub-agent, or an error message
        if execution fails.
    """

    args = ['-p', prompt]

    if model:
        args.extend(['-m', model])

    if sandbox:
        args.append('-s')

    # Collect only the final result from the stream
    for chunk in qwen_stream(args):
        if chunk.get('type') == 'result':
            # Return the final result
            return chunk.get('result', 'No result found in response')

    # If no result chunk was found, return an error message
    return "Error: No result received from Qwen"

if __name__ == "__main__":
    mcp.run()
