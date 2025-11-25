# Background Agent MCP Tool

A powerful MCP tool that allows Claude to spawn background tasks that run independently, mimicking the cloud agent functionality locally.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude (Main Session)                     │
│                                                               │
│  1. submit_background_task("complex task") → returns task_id │
│  2. check_task_status(task_id) → "running"                   │
│  3. get_task_result(task_id) → full response                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├── MCP Tool Call
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Background Agent MCP                        │
│                                                               │
│  ┌─────────────┐      ┌──────────────┐                       │
│  │   SQLite    │◄────►│  Task Queue  │                       │
│  │   Database  │      │  Management  │                       │
│  └─────────────┘      └──────────────┘                       │
│                              │                                │
│                              ├── Spawns Independent Processes │
│                              ▼                                │
│         ┌────────────────────────────────────┐               │
│         │  Background Process (daemon=False)  │               │
│         │  - Makes LLM API call               │               │
│         │  - Updates DB directly              │               │
│         │  - Survives MCP restart             │               │
│         └────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├── LLM Provider Abstraction
                              ▼
     ┌──────────────┬──────────────┬──────────────┐
     │    Claude    │    Gemini    │    Ollama    │
     │  (Anthropic) │   (Google)   │   (Local)    │
     └──────────────┴──────────────┴──────────────┘
```

## Why This Architecture?

### 1. **No nohup needed**
   - Uses `multiprocessing.Process` with `daemon=False`
   - Processes survive parent termination
   - Proper OS-level process isolation

### 2. **Persistent State**
   - SQLite database at `~/.cache/claude_background_agent/tasks.db`
   - Survives MCP server restarts
   - Query historical tasks anytime

### 3. **Environment Handling**
   - Each process inherits parent environment
   - API keys from env vars (ANTHROPIC_API_KEY, GOOGLE_API_KEY)
   - Isolated execution contexts

### 4. **Easy LLM Swapping**
   - Abstract `LLMProvider` interface
   - Add new providers by implementing `complete()` method
   - Switch models with simple string parameter

### 5. **Session Management**
   - Each task gets unique UUID
   - Track: pending → running → completed/failed
   - Full audit trail with timestamps

## Installation & Setup

### 1. Install Dependencies

```bash
# Using uv (as per your preference)
uv pip install fastmcp anthropic google-generativeai requests
```

### 2. Set Up API Keys

```bash
# For Claude
export ANTHROPIC_API_KEY="your-key-here"

# For Gemini
export GOOGLE_API_KEY="your-key-here"

# For Ollama (optional, defaults to localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"
```

### 3. Add to MCP Configuration

Edit your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "background-agent": {
      "command": "python",
      "args": ["/home/ntlpt24/agilab/src/mcp_tools/background_agent.py"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Usage Examples

### Example 1: Simple Background Task

```python
# In Claude session:
task_id = submit_background_task(
    task_description="Write a comprehensive guide on quantum computing including history, principles, and applications",
    model="claude-3-5-sonnet-20241022"
)

# Returns immediately with task_id
# Continue working on other things...

# Later, check status:
check_task_status(task_id)
# → "Status: running"

# When complete:
get_task_result(task_id)
# → Full response from Claude
```

### Example 2: Using Different Models

```python
# Use Claude Opus for complex reasoning
task1 = submit_background_task(
    "Analyze the economic implications of AI...",
    model="claude-3-opus-20240229"
)

# Use Gemini for different perspective
task2 = submit_background_task(
    "Summarize recent papers on...",
    model="gemini-pro"
)

# Use local Ollama for cost-free processing
task3 = submit_background_task(
    "Generate creative story ideas...",
    model="ollama:llama2"
)
```

### Example 3: Task Management

```python
# List all recent tasks
list_background_tasks(limit=10)

# List only completed tasks
list_background_tasks(status_filter="completed")

# Cancel a running task
cancel_task(task_id)
```

## Available Tools

### 1. `submit_background_task`
Spawns a background task that runs independently.

**Parameters:**
- `task_description` (str): The prompt to send to the LLM
- `model` (str): Model identifier (default: "claude-3-5-sonnet-20241022")
- `model_params` (dict): Optional params like `{"max_tokens": 8192}`

**Returns:** Task ID string

### 2. `check_task_status`
Check if a task is pending/running/completed/failed.

**Parameters:**
- `task_id` (str): The task ID

**Returns:** Status information with timestamps

### 3. `get_task_result`
Retrieve the output of a completed task.

**Parameters:**
- `task_id` (str): The task ID

**Returns:** The LLM's full response

### 4. `list_background_tasks`
View recent tasks with filtering.

**Parameters:**
- `limit` (int): Max number to show (default: 10)
- `status_filter` (str): Filter by status (optional)

**Returns:** List of tasks

### 5. `cancel_task`
Mark a task as cancelled.

**Parameters:**
- `task_id` (str): The task ID

**Returns:** Confirmation message

## Supported Models

### Claude (Anthropic)
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Gemini (Google)
- `gemini-pro`
- `gemini-pro-vision`

### Ollama (Local)
- `ollama:llama2`
- `ollama:mistral`
- `ollama:codellama`
- Any model you have installed locally

## Adding New LLM Providers

Easy to extend! Just inherit from `LLMProvider`:

```python
class CustomProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        # Your implementation here
        # Call your API, local model, etc.
        return response_text

# Then modify get_provider() to recognize your model string:
def get_provider(model: str) -> LLMProvider:
    if model.startswith("custom:"):
        return CustomProvider()
    # ...
```

## How It Works Internally

1. **Task Submission**:
   - Creates unique UUID
   - Inserts into SQLite with status="pending"
   - Spawns independent Python process
   - Returns immediately with task_id

2. **Background Execution**:
   - Process updates status to "running"
   - Makes LLM API call
   - Stores result in database
   - Updates status to "completed"
   - Process terminates

3. **Result Retrieval**:
   - Query SQLite by task_id
   - Return stored result
   - Works even if MCP server restarted

## Advantages Over Cloud Agent

| Feature | Cloud Agent | This Tool |
|---------|-------------|-----------|
| Cost | High (API costs) | Lower (your keys) |
| Privacy | Sends to cloud | Runs locally |
| Model Choice | Fixed | Any provider |
| Persistence | Cloud-dependent | Local SQLite |
| Offline | No | Yes (with local models) |

## Common Use Cases

1. **Long-running analysis** while you work on other things
2. **Parallel processing** of multiple tasks with different models
3. **Cost optimization** by using cheaper/local models for simple tasks
4. **Experimentation** with different models on same prompt
5. **Background research** that doesn't block your main workflow

## Troubleshooting

### Task stuck in "pending"
- Check if process actually started (look for Python processes)
- Check API keys are set correctly
- Check MCP server logs

### "API key not found"
- Ensure env vars are set in MCP config
- Restart Claude Desktop after config changes

### Task failed with error
- Use `check_task_status(task_id)` to see error message
- Common issues: API quota, network problems, invalid model name

## Future Enhancements

- [ ] Add task priorities
- [ ] Streaming results for long outputs
- [ ] Task dependencies (run task B after task A)
- [ ] Automatic retries on failure
- [ ] Web UI for task monitoring
- [ ] Resource limits per task

## Comparison with Other Approaches

### Why not just use `subprocess` with `nohup`?
- Less portable (platform-specific)
- Harder to track process state
- No built-in result storage
- Messy cleanup

### Why not use Celery/RQ?
- Overkill for single-user tool
- Requires Redis/broker setup
- More complex deployment
- We don't need distributed task queue

### Why SQLite instead of JSON files?
- Concurrent access safety
- ACID transactions
- Better querying
- Production-ready

## License & Credits

Built for personal use with FastMCP framework.
Compatible with Claude Code MCP ecosystem.
