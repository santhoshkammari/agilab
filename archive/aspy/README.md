# aspy - Advanced Structured Python for LLMs

**aspy** is a Python framework designed for building and managing language model (LM) based applications with structured signatures, batching capabilities, and evaluation tools. It provides a clean, efficient way to define and execute multi-stage language model programs with intelligent batching and dependency management.

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
git clone https://github.com/your-username/aspy.git
cd aspy

# Install as a package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import aspy as a

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
  aspy.Example(question="What is 2+2?", answer="4"),
  aspy.Example(question="Capital of France?", answer="Paris"),
]

# Create a module to evaluate
math_qa = aspy.Predict("question -> answer")

# Evaluate with progress bar and scoring
evaluator = aspy.Evaluate(
  devset=examples,
  metric=aspy.exact_match,
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
The base building block for aspy applications:

- `a.Module`: Base class for creating multi-stage programs
- `a.Predict`: Basic prediction module that maps inputs to outputs
- `a.ChainOfThought`: Reasoning-based module that shows its thinking process

### Language Model Integration
aspy supports various LM backends with intelligent batching:

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

## Project Structure

```
aspy/
├── __init__.py           # Main imports and configuration
├── batch_orchestrator.py # Batch execution and dependency management
├── evaluate/             # Evaluation framework
│   └── __init__.py       # Evaluation utilities and metrics
├── lm/                   # Language model interface
│   └── lm.py             # LM abstraction and batching
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