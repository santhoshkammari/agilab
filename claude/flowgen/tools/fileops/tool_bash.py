import subprocess
import shlex
import os
import time
from typing import Optional, Dict, Any


def bash_execute(command: str, description: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute a bash command in a persistent shell session with security measures and timeout controls.
    
    Args:
        command (str): The bash command to execute
        description (str, optional): 5-10 word description of what the command does
        timeout (int, optional): Timeout in milliseconds (default: 120000, max: 600000)
        
    Returns:
        Dict[str, Any]: Result containing:
            - success (bool): Whether command executed successfully
            - stdout (str): Standard output from command
            - stderr (str): Standard error from command
            - exit_code (int): Exit code of the command
            - execution_time (float): Time taken to execute in seconds
            - truncated (bool): Whether output was truncated
            
    Raises:
        ValueError: If command is empty, timeout is invalid, or paths are improperly quoted
        TimeoutError: If command exceeds timeout limit
    """
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")
    
    # Set default timeout and validate
    if timeout is None:
        timeout = 120000  # 2 minutes default
    
    if timeout < 1 or timeout > 600000:
        raise ValueError("Timeout must be between 1ms and 600000ms (10 minutes)")
    
    timeout_seconds = timeout / 1000.0
    
    # Validate command for common issues
    _validate_command(command)
    
    # Track execution time
    start_time = time.time()
    
    try:
        # Execute command with timeout
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable='/bin/bash'
        )
        
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        execution_time = time.time() - start_time
        raise TimeoutError(f"Command timed out after {timeout}ms: {command}")
    
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'success': False,
            'stdout': '',
            'stderr': f"Command execution failed: {str(e)}",
            'exit_code': -1,
            'execution_time': execution_time,
            'truncated': False
        }
    
    execution_time = time.time() - start_time
    
    # Check if output needs truncation (30,000 character limit)
    truncated = False
    max_output_length = 30000
    
    if len(stdout) > max_output_length:
        stdout = stdout[:max_output_length] + "\n... [OUTPUT TRUNCATED] ..."
        truncated = True
    
    if len(stderr) > max_output_length:
        stderr = stderr[:max_output_length] + "\n... [ERROR OUTPUT TRUNCATED] ..."
        truncated = True
    
    success = exit_code == 0
    
    return {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': execution_time,
        'truncated': truncated
    }


def _validate_command(command: str) -> None:
    """
    Validate command for common security and syntax issues.
    
    Args:
        command (str): Command to validate
        
    Raises:
        ValueError: If command has validation issues
    """
    # Check for prohibited commands that should use specialized tools
    prohibited_patterns = [
        ('find ', 'Use Glob tool instead of find command'),
        ('grep ', 'Use Grep tool instead of grep command (or rg for ripgrep)'),
        ('cat ', 'Use Read tool instead of cat command'),
        ('head ', 'Use Read tool instead of head command'),
        ('tail ', 'Use Read tool instead of tail command'),
        ('ls ', 'Use LS tool instead of ls command'),
    ]
    
    command_lower = command.lower().strip()
    for pattern, message in prohibited_patterns:
        if command_lower.startswith(pattern):
            raise ValueError(f"Prohibited command: {message}")
    
    # Check for interactive flags that won't work
    if ' -i ' in command or command.endswith(' -i'):
        raise ValueError("Interactive commands with -i flag are not supported")
    
    # Warning for paths that might need quoting (basic check)
    if ' /' in command and '"' not in command and "'" not in command:
        # Look for potential unquoted paths with spaces
        parts = command.split()
        for part in parts:
            if '/' in part and ' ' in part:
                raise ValueError(f"Path with spaces should be quoted: {part}")


def format_command_output(result: Dict[str, Any], command: str, description: Optional[str] = None) -> str:
    """
    Format the command execution result for display.
    
    Args:
        result (Dict[str, Any]): Result from execute_bash_command
        command (str): The original command
        description (str, optional): Description of the command
        
    Returns:
        str: Formatted output string
    """
    lines = []
    
    if description:
        lines.append(f"Command: {description}")
    
    lines.append(f"$ {command}")
    
    if result['success']:
        if result['stdout']:
            lines.append(result['stdout'])
        if result['stderr']:
            lines.append(f"[stderr]: {result['stderr']}")
    else:
        lines.append(f"[ERROR] Command failed with exit code {result['exit_code']}")
        if result['stderr']:
            lines.append(result['stderr'])
        if result['stdout']:
            lines.append(f"[stdout]: {result['stdout']}")
    
    if result['truncated']:
        lines.append("[Note: Output was truncated due to size]")
    
    lines.append(f"[Execution time: {result['execution_time']:.2f}s]")
    
    return '\n'.join(lines)


def bash_tool_wrapper(command: str, description: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Main wrapper function that combines execution and formatting.
    This is the primary function to be used by the tool system.
    
    Args:
        command (str): The bash command to execute
        description (str, optional): 5-10 word description of what the command does
        timeout (int, optional): Timeout in milliseconds
        
    Returns:
        str: Formatted command output
        
    Raises:
        ValueError: If command validation fails
        TimeoutError: If command times out
    """
    result = execute_bash_command(command, description, timeout)
    return format_command_output(result, command, description)