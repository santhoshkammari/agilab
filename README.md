# agilab

optimization experiments. building what's missing.

Frameworks don't work for real stuff. Built simple versions that do.

---

## What Got Built

| Thing | Purpose | Numbers |
|-------|---------|---------|
| **Metis** | Simple LLM/agent/eval framework | ~500 lines, no dependencies |
| **UniversalLogger** | One logger for everything | Handles strings, dicts, lists, AI chats, tables |
| **MCP Tools** | Tools for Claude Code | 3.6x faster, 94% cheaper web search |
| **TABLET** | Table recognition | F1: 0.72-0.81 (H/V splits) |
| **Tool SLM** | Small model for tool calls | Trained LFM2-350M with GRPO |

---

## Why Built These

### Metis
- Frameworks like LangChain too complicated
- Just needed simple async LLM calls
- Works with vLLM batching (23x speedup)

### UniversalLogger
- Normal logging needs too much setup
- Wanted one tool that handles all data types
- Zero config needed

### MCP Tools
**Web Search:**
- MCP version: 28s, $0.095
- Built-in: 102s, $1.515
- Result: 3.6x faster, 94% cheaper

**Other tools:**
- multigrep - search multiple patterns at once
- edit_operations - edit multiple lines/files
- gmail - send emails
- youtube tools

### TABLET
**Problem:** Original code used fixed 5px for all table gaps

**Fix:** Use actual gap width (2px, 10px, 50px, etc.)

**Results:**
```
Validation:  H-F1: 0.722  V-F1: 0.811
Training:    H-F1: 0.743  V-F1: 0.814
```

Works better for dense tables with tight spacing.

### Tool SLM
**Goal:** Train small model to call tools properly

**What we did:**
- Custom XML format (simpler than JSON)
- SFT then GRPO training
- Rewards shorter outputs

---

## Framework Analysis

Looked at: PydanticAI, LangChain, LlamaIndex, CrewAI, AutoGen

**Problems found:**
- Locked into paid services (Logfire, LangSmith)
- Too many abstractions
- APIs keep breaking
- Can't really see what's happening

**Decision:** Build minimal versions ourselves. ~1000 lines total that we understand.

---

## Structure

```
agilab/
├── exp/
│   ├── tablet/        # table recognition
│   ├── grpo_slm/      # tool model training
│   └── dra/           # arxiv fetcher
├── src/
│   ├── ai/            # metis framework
│   ├── logger/        # universal logger
│   └── mcp_tools/     # mcp tools
└── misc/              # analysis docs
```

---
