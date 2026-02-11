# Hierarchical Code Agent

A sophisticated multi-agent system for autonomous code generation and modification, inspired by the Cursor blog post on scaling AI coding agents.

## Architecture

The system implements a hierarchical structure with distinct agent roles:

```
MainPlanner (1)
    ↓
SubPlanners (N, parallel)
    ↓
Workers (M, concurrent)
    ↓
Judge (1)
```

### Agent Roles

1. **MainPlanner**
   - Explores entire codebase structure
   - Identifies 2-5 major domains/areas
   - Spawns SubPlanners for each domain
   - Creates high-level overview

2. **SubPlanner**
   - Explores specific domain in depth
   - Reads existing code
   - Breaks work into atomic tasks
   - Can spawn more SubPlanners recursively
   - Creates 3-10 tasks per domain

3. **Worker**
   - Grabs one task from queue
   - Completes it fully
   - Makes code changes
   - No coordination with other workers
   - Returns success/failure

4. **Judge**
   - Reviews cycle completion
   - Assesses progress toward goal
   - Decides: continue (fresh cycle) or stop
   - Provides reasoning

## Key Features

### Hierarchy Solves Coordination
- No flat structure with locks/coordination
- Clear responsibilities per role
- Parallel exploration and execution
- No bottlenecks

### Fresh Starts Combat Drift
- Each cycle starts fresh
- Planners re-explore actual codebase
- Memory = the code itself
- Recovers from mistakes automatically

### Parallel + Recursive
- SubPlanners run in parallel
- Can spawn more SubPlanners
- Workers execute concurrently
- Scales to hundreds of agents

### Database-Backed Task Queue
- SQLite for task management
- No file locks or shared state
- Atomic operations
- Persistent across cycles

## Algorithm

```python
LOOP:
  # Phase 1: Planning
  MainPlanner:
    - Explore entire codebase
    - Identify major areas
    - Spawn SubPlanners

  # Phase 2: Sub-planning (parallel)
  SubPlanners:
    - Explore specific domain
    - Break into atomic tasks
    - Add to task queue

  # Phase 3: Execution (concurrent)
  Workers:
    - Grab task from queue
    - Complete fully
    - Push to same branch
    - No coordination

  # Phase 4: Review
  Judge:
    - Review cycle
    - Decide: continue or stop

  # Phase 5: Fresh Start
  IF continue:
    Clear tasks
    LOOP (planners re-explore with new code)
  ELSE:
    Done
```

## Code Analysis Tools

The system provides comprehensive tools for code exploration:

- `list_files_recursive`: List files matching pattern
- `read_file_content`: Read file with optional line range
- `write_file_content`: Write content with backup
- `search_code_pattern`: Search for patterns in code
- `get_directory_structure`: Get directory tree
- `git_status`: Get git status
- `git_commit_and_push`: Commit and push changes

## Usage

### Basic Usage

```python
import asyncio
from src.ai import LM
from src.agents.code_agent import hierarchical_code_agent

async def main():
    # Initialize language model
    lm = LM(model="vllm:", api_base="http://localhost:8000")

    # Run hierarchical code agent
    results = await hierarchical_code_agent(
        goal="Add user authentication system with JWT tokens",
        root_path="./src",
        branch="feature/auth-system",
        lm=lm,
        max_cycles=5,
        max_workers=10,
        auto_commit=True
    )

    # Print results
    for i, result in enumerate(results, 1):
        print(f"Cycle {i}: {result.tasks_completed} tasks completed")
        print(f"Decision: {result.continue_next_cycle}")

asyncio.run(main())
```

### Command Line Usage

```bash
# Basic usage
python -m src.agents.code_agent \
    --goal "Implement REST API for user management" \
    --root-path ./src \
    --branch feature/user-api

# Advanced usage
python -m src.agents.code_agent \
    --goal "Migrate from SQLite to PostgreSQL" \
    --root-path ./src \
    --branch feature/postgres \
    --model "vllm:" \
    --api-base "http://192.168.1.100:8000" \
    --max-cycles 10 \
    --max-workers 20 \
    --no-auto-commit
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `goal` | str | required | High-level goal to achieve |
| `root_path` | str | "." | Root path of codebase |
| `branch` | str | "main" | Git branch to push to |
| `lm` | LM | required | Language model instance |
| `max_cycles` | int | 5 | Maximum cycles to run |
| `max_workers` | int | 10 | Max concurrent workers |
| `auto_commit` | bool | True | Auto-commit after each cycle |

## Database Schema

### Tasks Table
```sql
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL,  -- critical, high, medium, low
    status TEXT NOT NULL,     -- pending, in_progress, completed, failed
    file_paths TEXT NOT NULL, -- JSON array
    created_by TEXT NOT NULL,
    assigned_to TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
```

### Planners Table
```sql
CREATE TABLE planners (
    planner_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    file_patterns TEXT NOT NULL,
    parent_planner TEXT,
    sub_planners TEXT,
    tasks_created TEXT,
    created_at TEXT NOT NULL
);
```

### Cycles Table
```sql
CREATE TABLE cycles (
    cycle_number INTEGER PRIMARY KEY,
    tasks_completed INTEGER,
    tasks_failed INTEGER,
    sub_planners_spawned INTEGER,
    workers_active INTEGER,
    continue_next_cycle BOOLEAN,
    judge_reasoning TEXT,
    created_at TEXT NOT NULL
);
```

## Example Scenarios

### Scenario 1: New Feature Implementation
```python
# Goal: Implement complete authentication system
results = await hierarchical_code_agent(
    goal="""
    Implement user authentication system with:
    - JWT token generation and validation
    - Login/logout endpoints
    - Password hashing with bcrypt
    - Refresh token mechanism
    - User session management
    """,
    root_path="./src",
    branch="feature/auth",
    lm=lm,
    max_cycles=3
)
```

### Scenario 2: Refactoring
```python
# Goal: Refactor monolithic app to modular architecture
results = await hierarchical_code_agent(
    goal="""
    Refactor monolithic application into modular architecture:
    - Separate concerns into modules
    - Create clear interfaces between modules
    - Extract shared utilities
    - Update imports and dependencies
    """,
    root_path="./src",
    branch="refactor/modular",
    lm=lm,
    max_cycles=5,
    max_workers=15
)
```

### Scenario 3: Bug Fixes
```python
# Goal: Fix known bugs
results = await hierarchical_code_agent(
    goal="""
    Fix the following bugs:
    1. Race condition in async file handler
    2. Memory leak in cache implementation
    3. Incorrect error handling in API routes
    """,
    root_path="./src",
    branch="bugfix/critical-issues",
    lm=lm,
    max_cycles=2
)
```

### Scenario 4: Testing
```python
# Goal: Add comprehensive tests
results = await hierarchical_code_agent(
    goal="""
    Add comprehensive test coverage:
    - Unit tests for all core modules
    - Integration tests for API endpoints
    - E2E tests for critical flows
    - Achieve >80% coverage
    """,
    root_path="./tests",
    branch="feature/testing",
    lm=lm,
    max_cycles=4,
    max_workers=20
)
```

## Design Principles

### 1. Hierarchy over Coordination
- Don't use flat structure with locks
- Use clear hierarchy with roles
- Let structure solve coordination

### 2. Fresh Starts over Memory
- Each cycle starts fresh
- Planners explore actual code
- No accumulated context drift
- Self-correcting system

### 3. Parallelism at Every Level
- SubPlanners run in parallel
- Workers execute concurrently
- Database handles synchronization

### 4. Atomicity over Dependencies
- Tasks are independent
- No inter-task dependencies
- Workers don't coordinate
- Reduces complexity

### 5. The Codebase is the Memory
- Don't store external state
- Read actual files each cycle
- Git history tracks progress
- Source of truth is always fresh

## Comparison with Flat Approach

| Aspect | Flat Structure | Hierarchical Structure |
|--------|----------------|----------------------|
| Coordination | Locks, shared files | Hierarchy, roles |
| Bottlenecks | Frequent | Rare |
| Parallelism | Limited | Extensive |
| Drift | Common | Self-correcting |
| Complexity | High | Lower |
| Scalability | ~10 agents | Hundreds |
| Robustness | Brittle | Resilient |

## Results from Cursor Blog

- **Browser from scratch**: 1M LoC in 1 week
- **Solid to React migration**: 3 weeks, 266K+ lines changed
- **Hundreds of agents** working concurrently
- **Weeks of autonomous work** without intervention

## Debugging and Monitoring

### Logs
The system uses structured logging:

```python
from src.logger.logger import get_logger

logger = get_logger(__name__, level='DEBUG')
```

### Database Inspection
```python
import sqlite3

conn = sqlite3.connect("code_agent_tasks.db")
cursor = conn.cursor()

# Get pending tasks
cursor.execute("SELECT * FROM tasks WHERE status = 'pending'")
print(cursor.fetchall())

# Get cycle stats
cursor.execute("SELECT * FROM cycles ORDER BY cycle_number DESC")
print(cursor.fetchall())
```

### Progress Tracking
```python
# Results include detailed statistics
for result in results:
    print(f"Cycle {result.cycle_number}:")
    print(f"  Completed: {result.tasks_completed}")
    print(f"  Failed: {result.tasks_failed}")
    print(f"  Continue: {result.continue_next_cycle}")
    print(f"  Reasoning: {result.judge_reasoning}")
```

## Limitations and Considerations

1. **LM Quality**: System performance depends on LM capabilities
2. **Cost**: Running hundreds of workers can be expensive
3. **Time**: Complex goals may take multiple cycles
4. **Verification**: Manual review recommended for production code
5. **Git Conflicts**: Manual resolution if branch diverges

## Future Enhancements

1. **Specialized Models**: Use different models for different roles
2. **Better Tools**: AST analysis, type checking, linting
3. **Human-in-the-Loop**: Optional approval gates
4. **Metrics**: Code quality metrics and analysis
5. **Rollback**: Automatic rollback on test failures

## References

- Cursor Blog: "How We Scaled AI Coding Agents"
- Fresh starts combat drift and tunnel vision
- Hierarchy solves coordination problems
- Different models excel at different roles

## Contributing

When extending the system:

1. Maintain clear role separation
2. Keep tasks atomic
3. Use database for state
4. Add comprehensive logging
5. Test with simple goals first

## License

See repository license.
