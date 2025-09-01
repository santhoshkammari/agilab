# Agilab

> A comprehensive software engineering platform with AI-powered tools for modern development workflows.

## Overview

Agilab is an ambitious project aimed at saturating the software engineering domain with AI capabilities. Our platform integrates cutting-edge technologies to assist developers throughout the entire software development lifecycle.

## Our Goals

### Main Goal
**Saturate software engineering with AI capabilities** - Transform how software is built, tested, and maintained through intelligent automation.

### Core Focus Areas

- **Logger** - *Completed*
  - Comprehensive logging infrastructure for tracking development processes
  - Detailed activity monitoring and analysis

- **DeepResearch** - *In Progress*
  - Advanced research capabilities using AI agents
  - Automated literature review and knowledge synthesis

- **LLM Integration** - *In Progress*
  - Multi-model support for various development tasks
  - Seamless integration with different LLM providers

- **Agent Systems** - *Planned*
  - Autonomous development agents for code generation
  - Collaborative AI systems for complex problem solving

- **Coding Assistance** - *Planned*
  - Intelligent code completion and refactoring
  - Automated bug detection and fixing

- **Evaluations** - *Planned*
  - Automated testing and evaluation frameworks
  - Performance benchmarking and optimization

- **Synthetic Data** - *Planned*
  - AI-generated test data and scenarios
  - Simulation environments for software testing

## Project Structure

This repository contains multiple specialized modules:

### [Claude](./claude)
A powerful terminal-based chat application with integrated LLM capabilities and extensible tool support.

### [Scout](./scout)
Optimized search and research capabilities with multi-provider LLM support.

### Spotlight
(Coming soon) Code analysis and optimization tools.

### STS
(Coming soon) Software testing and simulation framework.


## Current Status

- **Completed**: Logger module
- **Active Development**: DeepResearch, LLM Integration
- **Planning Phase**: Agent Systems, Coding Assistance, Evaluations, Synthetic Data

## Vision

We're building a complete ecosystem where AI agents work alongside developers to:
- Accelerate development processes
- Improve code quality and maintainability
- Automate repetitive tasks
- Enhance decision-making through data-driven insights
- Reduce time-to-market for software products

## Contributing

This project is under active development. Contributions are welcome as we work toward our goal of saturating software engineering with AI capabilities.
=======
# FlowGen

A unified framework for LLM interactions, agent workflows, and RL training.

## Core Architecture & Philosophy

**Universal Abstraction Layer**: Creates a unified interface that works across different LLM providers, training methods, and RL environments. The `__call__` method pattern makes everything behave like Python functions.

**Auto-Detection Intelligence**: Each component automatically detects usage patterns:
- `llm(string)` → single inference
- `llm([strings])` → batch processing  
- `env(llm)` → dataset evaluation
- `agent >> chain` → pipeline composition

## Quick Start

```python
from flowgen.llm import LLM
from flowgen.agent import Agent

# Initialize LLM (works with any provider)
llm = LLM(base_url="http://localhost:8000")  # Local API server
# llm = LLM(provider="openai", api_key="...")  # OpenAI
# llm = LLM(provider="ollama", model="llama3")  # Ollama

# Basic usage
result = llm("What is 2+2?")
print(result["content"])  # "4"
print(result["think"])    # Internal reasoning (if available)

# Agent with tools
agent = Agent(llm, tools=[get_weather, calculate])
response = agent("What's the weather in NYC and what's 15*23?")
```

## Framework Components

### 1. LLM Module (`llm/`)

**Abstraction Power**:
- `BaseLLM` provides unified interface for vLLM, Ollama, OpenAI, etc.
- Tool conversion system transforms Python functions to OpenAI tool schemas
- Streaming/non-streaming handled transparently
- Runtime parameter overrides: `llm(text, tools=[...], format=schema)`

**Production Readiness**:
- Proper error handling and timeout management
- Batch processing with ThreadPoolExecutor
- Tool calling standardization across providers
- **Thinking extraction** for reasoning models with `<think>` tags
- Handles incomplete thinking blocks and token limits

### 2. Agent Module (`agent/`)

**Agentic Intelligence**:
- Wraps any LLM with autonomous tool-calling loops
- Rich debugging with formatted output panels
- History management and conversation continuity
- Agent chaining with `>>` operator
- Export/import for conversation persistence

**Flexibility**:
- Works with any BaseLLM implementation
- Streaming and non-streaming execution
- Async/sync compatibility
- Pluggable tool system

### 3. API Server (`api/`)

**Local LLM Server**:
- FastAPI-based server for llama.cpp models
- Supports both streaming and non-streaming responses
- Tool calling and JSON schema validation
- Compatible with OpenAI ChatCompletion format

**Usage**:
```python
# Start server
python -m flowgen.api.api

# Use with LLM client
llm = LLM(base_url="http://localhost:8000")
```

### 4. RL Environment (`rl/`)

**Universal RL Interface**:
- Works with any reward function signature
- Multi-turn tool-enabled environments
- Automatic trainer integration via `env >> trainer`
- HuggingFace dataset compatibility

**Training Integration**:
- GRPO, PPO, DPO trainer compatibility
- Reward function composition with weights
- Preference dataset generation capabilities

## Key Innovations

1. **Pythonic Simplicity**: Everything feels like native Python - no complex configuration files or verbose APIs
2. **Composability**: Components chain naturally:
   ```python
   result = (env >> trainer).train()
   agent1 >> agent2 >> agent3  # Pipeline
   ```
3. **Auto-Detection**: Framework infers intent from usage patterns, reducing cognitive load
4. **Loose Coupling**: Each component works independently but integrates seamlessly
5. **Production Scale**: Thread pools, error handling, timeouts, proper streaming

## Recent Fixes

- **Thinking Extraction**: Fixed handling of incomplete `<think>` blocks from reasoning models
- **Token Limits**: Increased default max_tokens from 100 to 1000 for better response quality
- **Content Separation**: Properly separates thinking content from actual responses

## Testing

Run the test suites to verify functionality:

```bash
# Basic LLM functionality
python test_llm_basic.py

# Agent with tools
python test_agent_basic.py

# Streaming capabilities
python test_llm_streaming.py
```

## Extensibility

This architecture pattern can be extended to:

**Data Processing**:
```python
processor = DataProcessor(transforms=[clean, tokenize])
result = processor(dataset)  # Auto-batch
```

**Model Evaluation**:
```python
evaluator = Evaluator(metrics=[accuracy, f1])
scores = evaluator(model, test_data)
```

**Workflow Orchestration**:
```python
pipeline = step1 >> step2 >> step3
result = pipeline(input_data)
```

The framework demonstrates how to build production-ready systems that are both powerful and intuitive - a rare combination in ML infrastructure.

