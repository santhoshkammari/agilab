# dspy - Advanced Structured Python for LLMs

**dspy** is a Python framework designed for building and managing language model (LM) based applications with structured signatures, batching capabilities, and evaluation tools. It provides a clean, efficient way to define and execute multi-stage language model programs with intelligent batching and dependency management.

## Features

- **Structured Signatures**: Define clear input/output specifications with type hints using a simple string-based syntax
- **Batch Execution**: Optimize performance with intelligent batching and dependency-aware execution
- **Module Architecture**: Create multi-stage programs with reusable LM modules
- **Evaluation Framework**: Built-in evaluation tools with metrics and progress tracking
- **Optimization Tools**: Prompt and instruction optimization capabilities
- **Chain of Thought Reasoning**: Built-in support for reasoning-based tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dspy.git
cd dspy

# Install as a package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import dspy as a

# Configure with your LM
lm = a.LM(api_base="http://your-llm-server:8000")
a.configure(lm=lm)

# Define a simple prediction module
math = a.Predict("question -> answer:float")

# Execute
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)
```

### Multi-stage Module

```python
class DraftArticle(a.Module):
    def __init__(self):
        super().__init__()
        self.build_outline = a.ChainOfThought("topic -> title, two_sections:[str]")
        self.draft_section = a.ChainOfThought("topic, section_heading -> content")

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []

        # Handle sections list
        if hasattr(outline, 'two_sections') and outline.two_sections:
            for heading in outline.two_sections:
                section = self.draft_section(topic=outline.title, section_heading=f"## {heading}")
                sections.append(section.content)

        return a.Prediction(title=outline.title, sections=sections)

# Use the multi-stage module
draft_article = DraftArticle()
article = draft_article(topic="World Cup 2002")
print(article)
```

### Evaluation

```python
# Create test examples
examples = [
  dspy.Example(question="What is 2+2?", answer="4"),
  dspy.Example(question="Capital of France?", answer="Paris"),
]

# Create a module to evaluate
math_qa = dspy.Predict("question -> answer")

# Evaluate with progress bar and scoring
evaluator = dspy.Evaluate(
  devset=examples,
  metric=dspy.exact_match,
  display_progress=True,
  save_as_json="results.json"
)

result = evaluator(math_qa)
print(f"Final score: {result.score}%")
```

## Core Components

### Signatures
Define the interface for your language model modules using a simple string format:

```python
signature = a.Signature("question -> answer")
# Or with types: "question, context: str -> answer: str, confidence: float"
```

### Modules
The base building block for dspy applications:

- `a.Module`: Base class for creating multi-stage programs
- `a.Predict`: Basic prediction module that maps inputs to outputs
- `a.ChainOfThought`: Reasoning-based module that shows its thinking process

### Language Model Integration
dspy supports various LM backends with intelligent batching:

```python
lm = a.LM(api_base="http://your-llm-server:8000", model="vllm:")
```

## Roadmap

### ✅ Core Functionality
- [x] Basic module system with `a.Module`, `a.Predict`, `a.ChainOfThought`
- [x] Signature parsing with type hints
- [x] Batch execution capabilities
- [x] Example-based evaluation framework
- [x] Dependency-aware orchestration

### ✅ Advanced Features
- [x] Multi-stage module dependency management
- [x] Batch processing with optimized execution
- [x] Chain of thought reasoning
- [x] Custom type registration system
- [x] Evaluation metrics and progress tracking

### 🔄 In Development
- [ ] Synthetic data generation tools
- [ ] Auto-prompt optimization
- [ ] MCP (Model Communication Protocol) integration
- [ ] Advanced optimization algorithms (MIPROv2, GEPA)
- [ ] More evaluation metrics and visualization tools

### 🚧 Planned Features
- [ ] Automated testing framework
- [ ] More optimization algorithms
- [ ] Enhanced batching with dynamic scheduling
- [ ] Model fine-tuning integration
- [ ] Distributed execution support
- [ ] GUI dashboard for monitoring and debugging
- [ ] More comprehensive documentation and examples

## Agent Framework

The `agent.py` module provides a streaming-based agent framework inspired by kosong's LLM abstraction layer.

### Key Features

- **Single-turn LLM generation** with async tool execution
- **Early tool execution optimization** - tools start executing while LLM streams remaining tool calls
- **Parallel tool execution** - multiple tools run concurrently
- **Deferred results** - tools execute asynchronously, results awaited on-demand
- **Production-optimized** - ~0.5-1s latency reduction for multi-tool scenarios

### Core Components

- `gen()` - Streaming function for low-level LLM response generation
- `step()` - High-level agent step with automatic async tool execution
- `StepResult` - Result container with async tool futures
- `ToolResult` - Individual tool execution result
- `_execute_tool()` - Internal async tool execution handler

### Usage Example

```python
from agent import step, tool_result_to_message

def get_weather(city: str, unit: str = "celsius"):
    """Get current weather for a city"""
    return f"Weather in {city}: 22 degrees {unit}"

# Multi-turn agent loop
history = [{"role": "user", "content": "What's the weather in London?"}]
tools = [get_weather]

# Turn 1: LLM decides to use tool
result = await step(lm, history, tools)
print(result.message)

# Wait for tool execution
tool_results = await result.tool_results()
history.append(result.message)
for tr in tool_results:
    history.append(tool_result_to_message(tr))

# Turn 2: LLM processes tool results and responds
result2 = await step(lm, history, tools)
# No more tool calls, conversation complete
```

### Performance Optimization

**Early Tool Execution** (enabled by default):
- Tools spawn immediately when arguments are complete
- First tool can execute while LLM streams second tool
- Reduces total latency by 5-20% for multi-tool requests

```python
# Use default behavior (early execution enabled)
result = await step(lm, history, tools)

# Disable if needed
result = await step(lm, history, tools, early_tool_execution=False)
```

## Project Structure

```
dspy/
├── __init__.py           # Main imports and configuration
├── agent.py              # Streaming agent framework with async tool execution
├── lm.py                 # Language model interface
├── agent_run_sample.py   # Example usage and integration patterns
├── batch_orchestrator.py # Batch execution and dependency management
├── evaluate/             # Evaluation framework
│   └── __init__.py       # Evaluation utilities and metrics
├── optimize/             # Optimization algorithms
│   ├── mipro.py          # MIPROv2 optimization
│   └── gepa.py           # GEPA optimization
├── predict/              # Prediction modules
│   └── predict.py        # Module and Predict classes
├── signature/            # Signature parsing
│   └── signature.py      # Signature compiler
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see the LICENSE file for details.