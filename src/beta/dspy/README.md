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
- [ ] MCP (Model Communication Protocol) integration
- [ ] More evaluation metrics and visualization tools

### 🚧 Planned Features
- [ ] Automated testing framework
- [ ] Enhanced batching with dynamic scheduling
- [ ] Model fine-tuning integration
- [ ] Distributed execution support
- [ ] GUI dashboard for monitoring and debugging
- [ ] More comprehensive documentation and examples

## Agent Framework

The `agent.py` module provides a streaming-based agent framework with three levels of abstraction.

### Three-Level API

1. **`gen()`** - Low-level streaming (yields text chunks and tool calls)
2. **`step()`** - Single LLM generation with async tool execution
3. **`agent()`** - Multi-turn loop with max iterations (NEW)

### Key Features

- **Parallel tool execution** - multiple tools run concurrently
- **Early tool execution optimization** - tools start executing while LLM streams
- **Async/await throughout** - fully asynchronous
- **Multi-turn loops** - agent() handles iteration automatically
- **Production-optimized** - ~0.5-1s latency reduction for multi-tool scenarios

### Quick Example

```python
from agent import agent
from lm import LM

def calculator(expr: str) -> str:
    """Calculate expression"""
    return str(eval(expr))

result = await agent(
    lm=LM(),
    initial_message="What is 2+2? Then multiply by 5.",
    tools=[calculator],
    max_iterations=5
)

print(result['final_response'])
```

### Multi-Turn Manual Loop

```python
from agent import step, tool_result_to_message

history = [{"role": "user", "content": "..."}]
tools = [calculator]

for i in range(max_iterations):
    result = await step(lm, history, tools)
    history.append(result.message)

    if result.tool_calls:
        tool_results = await result.tool_results()
        for tr in tool_results:
            history.append(tool_result_to_message(tr))
    else:
        break  # Done, no more tool calls
```

See `AgentReadme.md` for full API documentation.

---

## Evaluation Framework

The `eval.py` module provides a minimal, async evaluation system with three levels.

### Three-Level API

1. **`eval_example()`** - Single example evaluation
2. **`eval_stream()`** - Streaming results (real-time monitoring)
3. **`eval_batch()`** - Batch processing (sequential or parallel)

### Two Evaluation Modes

**Mode 1: Direct Evaluation**
```python
from eval import eval_batch
from example import Example

examples = [
    Example(question="2+2?", answer="4"),
    Example(question="5*5?", answer="25"),
]

def exact_match(example, prediction):
    return 1.0 if str(prediction) == example.answer else 0.0

result = await eval_batch(
    module_fn=my_module,
    examples=examples,
    metric=exact_match,
    parallel=True
)
print(f"Score: {result['score']:.1f}%")
```

**Mode 2: Agent Evaluation** (NEW)
```python
from eval import eval_batch
from lm import LM

# Evaluate an agent end-to-end
result = await eval_batch(
    module_fn=lambda: None,  # Not used with use_step=True
    examples=examples,
    metric=metric,
    use_step=True,
    lm=LM(),
    tools=[calculator, search],
    parallel=True
)
```

### Key Features

- **Flexible metrics** - any function(example, prediction) → float
- **Real-time monitoring** - stream results for early stopping
- **Parallel evaluation** - concurrent batch processing
- **Agent support** - evaluate agentic systems with tools
- **Progress tracking** - built-in progress bars

See `EvalReadme.md` for full API documentation.

## Project Structure

```
dspy/
├── agent.py                     # Agent framework (gen, step, agent)
├── agent_run_sample.py          # Agent examples (gen, step)
├── agent_loop_sample.py         # Agent loop examples (agent function)
├── eval.py                      # Evaluation framework
├── eval_run_sample.py           # Direct evaluation examples
├── eval_run_sample_with_step.py # Agent evaluation examples
├── example.py                   # Example data structure
├── lm.py                        # Language model interface
├── AgentReadme.md               # Agent framework documentation
├── EvalReadme.md                # Evaluation framework documentation
├── README.md                    # This file (main documentation)
└── predict.py                   # Legacy: Module and Predict classes
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see the LICENSE file for details.