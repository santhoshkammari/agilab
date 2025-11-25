"""
Background Agent MCP Tool

Spawns background tasks using 'claude --print --dangerously-skip-permissions'
"""

import os
import json
import sqlite3
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from fastmcp import FastMCP

# Database and output paths
DB_PATH = Path.home() / ".cache" / "claude_background_agent" / "tasks.db"
OUTPUT_DIR = Path.home() / ".cache" / "claude_background_agent" / "outputs"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("Background Agent")


def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            output_file TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        )
    """)
    conn.commit()
    conn.close()


@mcp.tool
def submit_background_task(task_description: str) -> str:
    """
    Submit a task to run in background using Claude CLI.

    Args:
        task_description: The prompt/task for Claude

    Returns:
        Task ID to track the task

    Example:
        task_id = submit_background_task("Analyze all Python files and create a report")
    """
    init_db()

    task_id = str(uuid.uuid4())
    output_file = OUTPUT_DIR / f"{task_id}.txt"

    # Insert into database
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (task_id, description, status, output_file, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, task_description, "running", str(output_file), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    # Start background process
    cmd = ["claude", "--print", "--dangerously-skip-permissions", task_description]
    with open(output_file, 'w') as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)

    return f"Task ID: {task_id}\nStatus: running\n\nCheck with: check_task_status('{task_id}')"


@mcp.tool
def check_task_status(task_id: str) -> str:
    """
    Check status of a background task.

    Args:
        task_id: Task ID from submit_background_task

    Returns:
        Current status
    """
    init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT status, output_file, description FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: Task '{task_id}' not found"

    status, output_file, description = row

    # Check if task completed by checking output file size
    if status == "running" and output_file and os.path.exists(output_file):
        size = os.path.getsize(output_file)
        if size > 100:  # Has some output, likely done
            cursor.execute(
                "UPDATE tasks SET status = ?, completed_at = ? WHERE task_id = ?",
                ("completed", datetime.now().isoformat(), task_id)
            )
            conn.commit()
            status = "completed"

    conn.close()

    result = f"Task: {description[:80]}...\nStatus: {status}\n"
    if status == "completed":
        result += f"\nGet result: get_task_result('{task_id}')"

    return result


@mcp.tool
def get_task_result(task_id: str) -> str:
    """
    Get result of a completed task.

    Args:
        task_id: Task ID from submit_background_task

    Returns:
        Task output
    """
    init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT status, output_file FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return f"Error: Task '{task_id}' not found"

    status, output_file = row

    if status == "running":
        return f"Task still running. Check: check_task_status('{task_id}')"

    if not output_file or not os.path.exists(output_file):
        return "Error: Output file not found"

    with open(output_file, 'r') as f:
        output = f.read()

    return f"RESULT:\n{'='*60}\n{output}"


@mcp.tool
def list_background_tasks(limit: int = 10) -> str:
    """
    List recent background tasks.

    Args:
        limit: Max tasks to show

    Returns:
        List of tasks
    """
    init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT task_id, description, status, created_at FROM tasks ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No tasks found"

    result = f"Tasks ({len(rows)}):\n\n"
    for task_id, desc, status, created in rows:
        icon = "✓" if status == "completed" else "⏳"
        result += f"{icon} {task_id[:8]}... | {status}\n   {desc[:60]}...\n   {created}\n\n"

    return result


if __name__ == "__main__":
    init_db()
    mcp.run()
