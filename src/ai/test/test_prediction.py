"""
Test DSPy-style Prediction object with field access
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai

# Configure LM
print("=== Configuring LM ===")
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1
)
ai.configure(lm)

# Test 1: Simple signature with field access
print("\n=== Test 1: Field Access (DSPy-style) ===")
pred = ai.Predict("query -> answer")
result = pred(query="What is 2+2?")

print(f"Type: {type(result)}")
print(f"result.answer: {result.answer}")
print(f"str(result): {str(result)}")
print(f"repr(result): {repr(result)}")

# Test 2: Multi-field signature
print("\n=== Test 2: Multi-field Signature ===")
classifier = ai.Predict("query -> classification, confidence")
result = classifier(query="Show me sales data")

print(f"Type: {type(result)}")
if hasattr(result, 'classification'):
    print(f"result.classification: {result.classification}")
if hasattr(result, 'confidence'):
    print(f"result.confidence: {result.confidence}")
print(f"str(result): {str(result)}")

# Test 3: Use in Module (DSPy-style)
print("\n=== Test 3: Module with Field Access ===")
class RAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification")
        self.answer = ai.Predict("context, query -> answer")

    def forward(self, query):
        # Access .classification field (DSPy-style)
        c = self.classify(query=query)
        print(f"  Classification: {c.classification}")

        # Mock context based on classification
        if "sql" in c.classification.lower():
            context = "SQL data"
        else:
            context = "Vector data"

        # Access .answer field
        ans = self.answer(context=context, query=query)
        print(f"  Answer: {ans.answer}")

        return ans.answer  # Return the field value

pipeline = RAG()
answer = pipeline(query="Show sales numbers")
print(f"Final: {answer}")

# Test 4: Postprocessing with Prediction object
print("\n=== Test 4: Postprocessing ===")
def uppercase_answer(pred):
    # pred is a Prediction object
    return pred.answer.upper()

pred_post = ai.Predict("query -> answer", postprocess=uppercase_answer)
result = pred_post(query="Say hello")
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test 5: Raw mode (no signature)
print("\n=== Test 5: Raw Mode (No Signature) ===")
raw = ai.Predict(system="You are helpful")
result = raw(input="What is Python?")
print(f"Type: {type(result)}")
print(f"Has 'text' attr: {hasattr(result, 'text')}")
print(f"Result: {str(result)[:80]}...")

print("\n" + "="*60)
print("✅ All Prediction tests completed!")
print("="*60)
