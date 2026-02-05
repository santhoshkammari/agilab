# Changelog

## v2.0.0 - Production-Ready DSPy Integration (2024-02-06)

### Major Changes

#### New Features
- **DSPy Signature Integration**: Reuses DSPy's `Signature`, `ChatAdapter`, and `ensure_signature`
  - Supports string signatures: `"query -> answer"`
  - Supports custom Signature classes
  - Automatic field formatting via ChatAdapter

- **Module Base Class**: New `ai.Module` for composable pipelines
  - `forward()` method for defining pipeline logic
  - `named_predictors()` to list all Predict instances
  - `inspect_history()` for debugging
  - `reset()` to clear all histories

- **Flexible Prompting Modes**: Three modes in one `Predict` class
  1. DSPy Signature: `Predict("query -> answer")`
  2. Signature + System: `Predict("query -> answer", system="Be concise")`
  3. Raw System: `Predict(system="You are...")`

- **Conversation History Support**:
  - Pass `history` parameter: `pred(query="...", history=[...])`
  - History injected between system and user messages
  - Enables multi-turn conversations with context

- **Post-Processing**:
  - `postprocess` parameter for automatic output transformation
  - Example: `postprocess=lambda pred: json.loads(pred.text)`

- **Enhanced Predict Parameters**:
  - `signature`: DSPy signature (str or class)
  - `system`: Additional system prompt
  - `instructions`: Override signature instructions
  - `postprocess`: Post-processing function
  - All existing parameters maintained (lm, tools, max_iterations, etc.)

#### API Changes
- `Predict.__init__()`: Added `signature`, `system`, `instructions`, `postprocess`
- `Predict.__call__()`: Added `history` parameter
- `Predict.run()`: Added `history` parameter, signature field separation
- New exports: `Signature`, `InputField`, `OutputField` (from DSPy)

#### Internal Changes
- Added `_adapter` and `_signature_obj` to Predict for DSPy integration
- Separated signature fields from LLM parameters in kwargs
- Fixed indentation issues in LM class methods

### Migration Guide

#### From v1.x to v2.0

**No breaking changes** - all v1.x code still works!

New capabilities available:

```python
# v1.x style (still works)
pred = ai.Predict()
result = pred("What is 2+2?")

# v2.0 with signature
pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")

# v2.0 with history
result = pred(query="What's my name?", history=[...])

# v2.0 with Module
class Pipeline(ai.Module):
    def __init__(self):
        super().__init__()
        self.step1 = ai.Predict("query -> answer")

    def forward(self, query):
        return self.step1(query=query)
```

### Examples Added
- `test_production.py`: Comprehensive test suite
- `example_rag.py`: Production RAG pipeline
- `example_modifications.py`: Manager change scenarios

### Documentation
- New comprehensive README.md
- API reference with all parameters
- Real-world examples
- Design principles explained

### Testing
- All tests pass with real LLM
- Verified signature parsing
- Verified history injection
- Verified postprocessing
- Verified Module composition

---

## v1.x - Original Implementation

- Basic `LM` client with streaming
- `Predict` class with tool support
- `Eval` with caching and parallelization
- Async/sync support
- Parameter inheritance (LM → Predict → call)
