# Agilab

AI/LLM experimentation workspace with agents, tools, and utilities.

## Structure

- `src/ai/` - LLM framework (ai.py) with DSPy-style signatures
- `src/agents/` - Specialized agents (DRA, ArXiv, Scout, Search, DB)
- `src/mcp_tools/` - MCP tools for file operations, web, git, etc.
- `src/logger/` - Logging utilities
- `exp/` - Experiments (tablet splitting, arxiv, etc.)
- `misc/` - Various utilities and configs

## Quick Start

```python
from src.ai import ai

lm = ai.LM(model="Qwen/Qwen3-4B", api_base="http://localhost:8000")
ai.configure(lm)

pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")
print(result)  # "4"
```

## Requirements

- Python 3.10+
- dspy
- aiohttp
- transformers

## Notes

- Individual components have their own READMEs
- Use VPN to access internal GPU servers
