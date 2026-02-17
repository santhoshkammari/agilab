# Core Philosophy: Token & Time Efficiency
Every tool choice must minimize token cost and task completion time. These custom MCP tools exist for that exact purpose — they are the human shortcuts of Claude Code. Always prefer them.

## Tool Priority (Fastest & Cheapest First)
- **FastEdits (`mcp__FastEdits__edit`)** — batch multi-range edits in one call, no string matching overhead
- **FileEditOperations** — use `move_lines`, `copy_paste_*`, `replace_pattern_occurrences`, `remove_lines_by_pattern` for structural file ops
- **MultiRead/MultiEdit/MultiGrep MCP tools** — always try first; fallback to basic Read/Edit/Grep only after 2 failed attempts
- **Basic Edit/Read/Grep** — last resort only

## Package Management
- Use `uv` over `pip` — faster installs. Usage: `uv pip install xxx`

## Rules
- Never waste tokens reproducing large text blocks when line numbers suffice
- Batch edits into single tool calls wherever possible
- Always pick the tool with the lowest token + time cost for the job
