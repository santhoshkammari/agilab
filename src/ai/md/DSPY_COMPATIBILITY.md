# DSPy Compatibility Guide

## Field Access (The DSPy Way)

When you use a signature like `"query -> answer"`, the result is a `Prediction` object where you access fields by name:

```python
# DSPy-style field access
pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")

print(result.answer)  # "4" - Access by field name ✅
print(str(result))    # "4" - Convert to string
```

## Single Output Field

```python
classifier = ai.Predict("query -> classification")
result = classifier(query="Show me sales")

# Access the output field
print(result.classification)  # "SQL"
```

## Multiple Output Fields

```python
analyzer = ai.Predict("text -> sentiment, confidence")
result = analyzer(text="I love this!")

# Access each field
print(result.sentiment)    # "positive"
print(result.confidence)   # "high"
print(str(result))         # "positive" (first field)
```

## Using in Modules (The Right Way)

```python
class RAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification")
        self.answer = ai.Predict("context, query -> answer")

    def forward(self, query):
        # Access fields like DSPy ✅
        c = self.classify(query=query)
        print(f"Type: {c.classification}")

        # Use field values
        if "sql" in c.classification.lower():
            context = self.sql_search(query)
        else:
            context = self.vector_search(query)

        # Access answer field
        ans = self.answer(context=context, query=query)
        return ans.answer  # Return the field value
```

## Raw Mode (No Signature)

When you don't use a signature, you still get a Prediction object:

```python
pred = ai.Predict(system="You are helpful")
result = pred(input="Hello")

# Has .text attribute
print(result.text)    # The response text
print(str(result))    # Same as .text
```

## Postprocessing with Prediction

```python
def extract_number(pred):
    # pred is a Prediction object
    import re
    match = re.search(r'\d+', pred.answer)
    return match.group(0) if match else pred.answer

pred = ai.Predict("query -> answer", postprocess=extract_number)
result = pred(query="What is 2+2?")
print(result)  # "4" (extracted number)
```

## Comparison: ai.py vs DSPy

### ai.py (Our Implementation)

```python
import ai

lm = ai.LM(api_base="http://localhost:8000")
ai.configure(lm)

pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")
print(result.answer)  # "4"
```

### Standard DSPy

```python
import dspy

lm = dspy.OpenAI(api_base="http://localhost:8000")
dspy.configure(lm=lm)

pred = dspy.Predict("query -> answer")
result = pred(query="What is 2+2?")
print(result.answer)  # "4"
```

### Key Differences

| Feature | ai.py | DSPy |
|---------|-------|------|
| Field access | ✅ `result.answer` | ✅ `result.answer` |
| Multiple fields | ✅ `result.field1, result.field2` | ✅ Same |
| History support | ✅ Built-in `history=` param | ❌ Need manual handling |
| System prompts | ✅ `system=` param | ❌ Only via signature |
| Postprocessing | ✅ `postprocess=` param | ❌ Manual |
| Module pattern | ✅ `ai.Module` | ✅ `dspy.Module` |
| Optimizers | ❌ Not included | ✅ MIPRO, Bootstrap, etc. |

## Migration from String Returns

If you were using ai.py before and returning strings:

### Before (v1.x - returned strings)
```python
pred = ai.Predict()
result = pred("What is 2+2?")
print(result)  # "4" (string)
```

### After (v2.0 - returns Prediction)
```python
pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")
print(result.answer)  # "4" - Field access ✅
print(str(result))    # "4" - String conversion ✅
```

**Backward Compatibility**: Just use `str(result)` if you want the old behavior!

## Best Practices

### ✅ DO: Use field access (DSPy-style)

```python
pred = ai.Predict("query -> answer")
result = pred(query="...")
print(result.answer)  # Access by field name
```

### ✅ DO: Use in pipelines

```python
class Pipeline(ai.Module):
    def forward(self, query):
        c = self.classify(query=query)
        # Use c.classification directly
        return self.answer(query=query, category=c.classification)
```

### ❌ DON'T: Just print the result

```python
# This works but loses DSPy compatibility
result = pred(query="...")
print(result)  # Prints the whole object

# Better:
print(result.answer)  # Access the field explicitly
```

## Advanced: Custom Signatures

You can also use DSPy Signature classes:

```python
from dspy import Signature, InputField, OutputField

class Analyze(Signature):
    """Analyze text for sentiment and topics."""
    text = InputField(desc="Text to analyze")
    sentiment = OutputField(desc="positive, negative, or neutral")
    topics = OutputField(desc="Comma-separated list of topics")

pred = ai.Predict(Analyze)
result = pred(text="I love this new feature!")

print(result.sentiment)  # "positive"
print(result.topics)     # "feature, satisfaction"
```

## Summary

✅ **Use `result.fieldname` to access outputs** (DSPy-style)
✅ **Works with single or multiple output fields**
✅ **Use `str(result)` for backward compatibility**
✅ **Postprocessing functions receive Prediction objects**
✅ **Module pipelines work exactly like DSPy**

This makes ai.py a drop-in enhancement to DSPy with extra features (history, system prompts, postprocessing) while maintaining full compatibility!
