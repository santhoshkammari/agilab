# ai.py - Production-Ready LLM Framework

**Minimal, flexible, DSPy-style framework for production LLM applications.**

Built for teams where requirements change every 2 hours. Reuses DSPy's battle-tested Signature system while adding production essentials: conversation history, post-processing, and flexible prompting modes.

## Why ai.py?

✅ **Manager-Friendly**: Change pipeline steps in minutes, not hours
✅ **DSPy-Compatible**: Reuses proven Signature system
✅ **History-Aware**: Built-in conversation context
✅ **Flexible**: Multiple prompting modes (signature, system, raw)
✅ **Production-Ready**: Timeouts, error handling, evaluation
✅ **Simple Logging**: Just use print statements

## Quick Start

```python
import ai

# Configure once
lm = ai.LM(
    model="your-model",
    api_base="http://localhost:8000",
    temperature=0.1
)
ai.configure(lm)

# Simple usage
pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")
print(result)  # "4"
```

## Core Concepts

### 1. LM - Language Model Client

Simple OpenAI-compatible client with parameter inheritance.

```python
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1,  # LM default
    seed=42
)
```

**Parameter Priority**:
1. Call-specific: `pred(query="...", temperature=0.9)`
2. Predict-level: `Predict(signature, temperature=0.5)`
3. LM-level: `LM(temperature=0.1)`

### 2. Predict - Flexible Predictor

Three usage modes in one class:

#### Mode 1: DSPy Signature
```python
pred = ai.Predict("query -> answer")
result = pred(query="What is Python?")
```

#### Mode 2: Signature + System Prompt
```python
classifier = ai.Predict(
    "query -> classification",
    system="You are an expert classifier. Return only: SQL or VECTOR"
)
result = classifier(query="Show me sales data")
```

#### Mode 3: Raw System Prompt
```python
pred = ai.Predict(system="You are a helpful assistant. Be concise.")
result = pred(input="Explain Python")
```

### 3. Conversation History

```python
pred = ai.Predict("query -> answer")

history = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hello Alice!"}
]

result = pred(query="What's my name?", history=history)
# "Your name is Alice"
```

### 4. Post-Processing

```python
import json

def extract_json(pred):
    return json.loads(pred.text.strip())

pred = ai.Predict(
    "text -> result",
    system="Extract JSON from text",
    postprocess=extract_json
)

data = pred(text='The data is {"name": "Alice"}')
print(data["name"])  # "Alice" (already parsed!)
```

### 5. Module - Composable Pipelines

Build multi-step LLM pipelines that are easy to modify.

```python
class RAGPipeline(ai.Module):
    def __init__(self):
        super().__init__()

        # Define all LLM steps
        self.classify = ai.Predict(
            "query -> classification",
            system="Classify as SQL or VECTOR"
        )

        self.refine = ai.Predict(
            "query, classification -> refined_query"
        )

        self.answer = ai.Predict(
            "context, query -> answer"
        )

    def forward(self, query):
        # Easy to rearrange, add, or remove steps!
        c = self.classify(query=query)
        print(f"[Classify] {c}")

        r = self.refine(query=query, classification=c)
        print(f"[Refine] {r}")

        context = self.search(r)  # Your search logic
        ans = self.answer(context=context, query=r)
        print(f"[Answer] {ans}")

        return ans

    def search(self, query):
        # Your retrieval logic
        return "mock results"

# Usage
pipeline = RAGPipeline()
answer = pipeline(query="Show me Q4 sales")
```

**When requirements change:**
- Remove step: Comment out 2-3 lines
- Reorder: Cut & paste
- Add validation: Insert new Predict + if statement

### 6. Evaluation

Compare different prompts/approaches easily.

```python
def exact_match(example, prediction):
    return prediction.strip() == example['expected']

dataset = [
    {'input': 'What is 2+2?', 'expected': '4'},
    {'input': 'What is 3+3?', 'expected': '6'}
]

# Test approach A
pred_a = ai.Predict("query -> answer", system="Be concise")
eval_a = ai.Eval(exact_match, dataset, save_path="results_a.json")
result_a = eval_a(pred_a)
print(f"Approach A: {result_a['score']}%")

# Test approach B
pred_b = ai.Predict("query -> answer", system="Answer with just the number")
eval_b = ai.Eval(exact_match, dataset, save_path="results_b.json")
result_b = eval_b(pred_b)
print(f"Approach B: {result_b['score']}%")
print(f"Improvement: {result_b['score'] - result_a['score']}%")
```

## Real-World Examples

### Example 1: Query Classifier

```python
classifier = ai.Predict(
    "query -> classification",
    system="""Expert query classifier.
- SQL: structured data queries (sales, metrics, counts)
- VECTOR: semantic search (reviews, feedback, descriptions)
Return ONLY: SQL or VECTOR"""
)

result = classifier(query="Show me sales from last month")
print(result)  # "SQL"

result = classifier(query="What do customers think?")
print(result)  # "VECTOR"
```

### Example 2: Conversation with Memory

```python
chatbot = ai.Predict("query -> answer", system="You are a helpful assistant")

history = []

# First turn
response = chatbot(query="My name is Alice", history=history)
history.append({"role": "user", "content": "My name is Alice"})
history.append({"role": "assistant", "content": response})

# Second turn (remembers Alice)
response = chatbot(query="What's my name?", history=history)
print(response)  # "Your name is Alice"
```

### Example 3: Full RAG Pipeline

See `example_rag.py` for a complete, production-ready RAG system with:
- Query classification
- Query refinement
- Context retrieval
- Answer generation
- Validation with chat history

### Example 4: Rapid Modifications

See `example_modifications.py` for scenarios showing how to:
- Remove pipeline steps (< 1 minute)
- Reorder steps (< 1 minute)
- Change prompts (< 2 minutes)
- Add logging (< 2 minutes)
- A/B test approaches (< 5 minutes)

## API Reference

### LM

```python
lm = ai.LM(
    model: str = "",
    api_base: str = "http://localhost:8000",
    api_key: str = "-",
    timeout: Optional[aiohttp.ClientTimeout] = None,
    **defaults  # temperature, seed, max_tokens, etc.
)
```

### Predict

```python
pred = ai.Predict(
    signature: Optional[Union[str, type]] = None,  # DSPy signature
    system: Optional[str] = None,                   # System prompt
    instructions: Optional[str] = None,             # Override signature instructions
    lm: Optional[LM] = None,                        # Optional LM instance
    tools: Optional[list[Callable]] = None,         # Tool functions
    postprocess: Optional[Callable] = None,         # Post-processing function
    max_iterations: int = 10,
    **defaults  # temperature, seed, etc.
)

# Call
result = pred(
    input: Optional[str] = None,              # For raw mode
    history: Optional[list[dict]] = None,     # Conversation history
    system: str = "",                          # Override system prompt
    return_result: bool = False,               # Return AgentResult object
    **kwargs  # Signature fields OR LLM params
)
```

### Module

```python
class YourPipeline(ai.Module):
    def __init__(self):
        super().__init__()
        # Define Predict instances

    def forward(self, **kwargs):
        # Your pipeline logic
        pass

# Methods:
pipeline.named_predictors()           # List all Predict instances
pipeline.inspect_history("pred_name") # View predictor history
pipeline.reset()                       # Clear all histories
```

### Eval

```python
evaluator = ai.Eval(
    metric: Callable,              # (example, prediction) -> score
    dataset: list[dict],           # [{'input': ..., 'expected': ...}, ...]
    save_path: Optional[str] = None,  # Enable caching
    parallel: int = 1,             # Thread count
    batch_size: int = 10,          # Save frequency
    show_progress: bool = True
)

result = evaluator(predict_instance)  # Returns {'score': 85.5, 'correct': 17, 'total': 20, ...}
```

## Installation

```bash
pip install dspy aiohttp transformers
```

## Design Principles

1. **Reuse, Don't Reinvent**: Uses DSPy's Signature system
2. **Flexibility First**: Multiple prompting modes, easy to extend
3. **Manager-Friendly**: Easy to modify when requirements change
4. **Production-Ready**: Proper error handling, timeouts, evaluation
5. **Simple Logging**: Just use print statements, no fancy frameworks
6. **Minimal Dependencies**: Only DSPy, aiohttp, transformers

## When to Use ai.py vs. DSPy

**Use ai.py when:**
- Requirements change frequently (every few hours/days)
- Need conversation history support
- Want flexible prompting (signature + system)
- Need simple, visible logging (print-based)
- Building RAG/agent systems with multiple LLM calls

**Use DSPy when:**
- Optimizing prompts automatically with few-shot examples
- Need advanced optimization (MIPRO, BootstrapFewShot)
- Building complex reasoning systems (Chain-of-Thought)

**Use both together:**
- ai.py for fast experimentation and development
- DSPy optimizers when you're ready to optimize
- Both share the same Signature system!

## Testing

```bash
# Run basic tests
python -c "import ai; print('✓ Import successful')"

# Run production tests (requires running LLM server)
python test_production.py

# Run examples
python example_rag.py
python example_modifications.py
```

## Files

- `ai.py` - Main implementation (~700 lines)
- `test_production.py` - Production tests with LLM
- `example_rag.py` - Complete RAG pipeline example
- `example_modifications.py` - Manager change scenarios
- `README.md` - This file

## License

MIT

## Contributing

Issues and PRs welcome! This framework is designed to be:
- Minimal (< 700 lines)
- Easy to modify
- Production-ready
- Manager-friendly

Keep it simple!
