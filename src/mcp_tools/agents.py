from typing import Dict, Any, List
import json
import uuid
import os
import subprocess
import sys
from fastmcp import FastMCP

mcp = FastMCP("Run-Agents")


@mcp.tool
def run_agents_manifest(path: str) -> Dict[str, Any]:
    """
    Load and execute a list of agent specifications from an agents.json file.

    This function is designed to be exposed as a TOOL to an LLM-based orchestrator
    (e.g., Claude). The tool accepts only a file path and performs the following steps:

    1. Reads the agents.json file at the given path.
    2. Validates that the file contains a list of agent specifications.
    3. Executes each agent synchronously using subprocess.
    4. Returns a run identifier and execution results.

    The agents.json file MUST be a JSON array where each item has the form:
        {
          "id": "string (optional but recommended)",
          "description": "string (optional)",
          "system_prompt": "string (required)",
          "task": "string (optional) - if present, will be appended to the system_prompt to create the full instruction",
          "tools": ["tool_name", ...] (optional),
          "model": "model_name" (optional)
        }

    Important constraints:
    - This tool executes agents synchronously and waits for completion.
    - All execution intent must be encoded inside agents.json.
    - Agents will be executed using qwen subprocess with current directory context.

    Parameters
    ----------
    path : str
        Path to the agents.json file. Must exist and be readable.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - run_id (str): Unique identifier for this execution run
        - status (str): One of ["started", "validation_failed", "completed"]
        - agents (List[Dict]): Execution results for each agent
        - errors (List[str]): Validation or execution errors if any occurred

    Example
    -------
    >>> run_agents_manifest("./agents.json")
    {
        "run_id": "run-8c1f...",
        "status": "completed",
        "agents": [
            {"id": "agent-1", "state": "completed", "result": "..."},
            {"id": "agent-2", "state": "completed", "result": "..."}
        ],
        "errors": []
    }

    With 'task' key:
    >>> run_agents_manifest("./agents.json")
    {
        "run_id": "run-8c1f...",
        "status": "completed",
        "agents": [
            {"id": "agent-with-task", "state": "completed", "result": "..."}
        ],
        "errors": []
    }

    Sample agents.json with task:
    [
        {
          "id": "my-agent",
          "description": "Sample agent with task",
          "system_prompt": "You are a helpful assistant.",
          "task": "Perform a specific action like running a command or processing data.",
          "tools": ["run_shell_command"],
          "model": "default"
        }
    ]
    """
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    errors: List[str] = []

    # --- Step 1: Load file ---
    if not os.path.isfile(path):
        return {
            "run_id": None,
            "status": "validation_failed",
            "agents": [],
            "errors": [f"agents.json not found at path: {path}"]
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            agents_data = json.load(f)
    except Exception as e:
        return {
            "run_id": None,
            "status": "validation_failed",
            "agents": [],
            "errors": [f"Failed to read agents.json: {str(e)}"]
        }

    # --- Step 2: Validate structure ---
    if not isinstance(agents_data, list):
        return {
            "run_id": None,
            "status": "validation_failed",
            "agents": [],
            "errors": ["agents.json must be a JSON array"]
        }

    # --- Step 3: Execute agents synchronously ---
    executed_agents = []

    for idx, agent in enumerate(agents_data):
        if not isinstance(agent, dict):
            errors.append(f"Agent at index {idx} is not an object")
            continue

        if "system_prompt" not in agent or not isinstance(agent["system_prompt"], str):
            errors.append(f"Agent at index {idx} is missing required field 'system_prompt'")
            continue

        agent_id = agent.get("id", f"agent-{idx}")

        # Prepare the agent execution
        # Using subprocess to run the qwen command with the system prompt
        try:
            # Get the current directory to preserve the working context
            current_dir = os.getcwd()

            # Extract agent config and check for task
            agent_config = json.loads(json.dumps(agent))
            system_prompt = agent_config['system_prompt']

            # If there's a 'task' key in the agent config, add it to the system prompt
            task = agent_config.get('task', '')
            if task:
                # Append the task to the system prompt to make it part of the instruction
                full_prompt = f"{system_prompt} The specific task you need to perform is: {task}"
            else:
                full_prompt = system_prompt

            # Prepare the command arguments for the qwen subprocess
            cmd = [sys.executable, "-c", f"""
import subprocess
import sys
import json
import os

full_prompt = {json.dumps(full_prompt)}

# Prepare arguments for qwen command
# Using -y flag (yolo mode) to automatically accept all actions
args = ["qwen", "-p", full_prompt, "-y"]

try:
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    output = result.stdout
    if result.stderr:
        output += "\\nSTDERR: " + result.stderr
    print(output)
except subprocess.TimeoutExpired:
    print("ERROR: Agent execution timed out after 300 seconds")
except Exception as e:
    print(f"ERROR: Failed to execute agent: {{str(e)}}")
"""]

            # Execute the command in the current directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)

            agent_result = {
                "id": agent_id,
                "state": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            executed_agents.append(agent_result)

        except Exception as e:
            errors.append(f"Failed to execute agent {agent_id}: {str(e)}")
            executed_agents.append({
                "id": agent_id,
                "state": "failed",
                "error": str(e)
            })

    if errors:
        return {
            "run_id": run_id,
            "status": "completed_with_errors",
            "agents": executed_agents,
            "errors": errors
        }

    # --- Step 4: Return execution results ---
    return {
        "run_id": run_id,
        "status": "completed",
        "agents": executed_agents,
        "errors": []
    }

if __name__=="__main__":
    mcp.run()
