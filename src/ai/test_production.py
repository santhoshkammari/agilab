"""
Test production-ready ai.py implementation with all features.
"""
import ai

# Configure LM
print("=== Configuring LM ===")
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1
)
ai.configure(lm)
print("✓ LM configured")

# Test 1: Simple signature
print("\n=== Test 1: Simple Signature ===")
pred = ai.Predict("query -> answer")
try:
    result = pred(query="What is 2+2?")
    print(f"Result: {result}")
    print("✓ Test 1 passed")
except Exception as e:
    print(f"❌ Test 1 failed: {e}")

# Test 2: Signature + System Prompt
print("\n=== Test 2: Signature + System ===")
classifier = ai.Predict(
    "query -> classification",
    system="You are a classifier. Return only: SQL or VECTOR"
)
try:
    result = classifier(query="Show me sales data")
    print(f"Classification: {result}")
    print("✓ Test 2 passed")
except Exception as e:
    print(f"❌ Test 2 failed: {e}")

# Test 3: With History
print("\n=== Test 3: With History ===")
history = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hi Alice!"}
]
pred_history = ai.Predict("query -> answer")
try:
    result = pred_history(query="What's my name?", history=history)
    print(f"Result: {result}")
    print("✓ Test 3 passed")
except Exception as e:
    print(f"❌ Test 3 failed: {e}")

# Test 4: Postprocessing
print("\n=== Test 4: Postprocessing ===")
def extract_number(pred):
    import re
    match = re.search(r'\d+', pred.text)
    return match.group(0) if match else pred.text

pred_post = ai.Predict("query -> answer", postprocess=extract_number)
try:
    result = pred_post(query="What is 2+2? Answer with just the number.")
    print(f"Extracted: {result}")
    print("✓ Test 4 passed")
except Exception as e:
    print(f"❌ Test 4 failed: {e}")

# Test 5: Raw System Prompt (no signature)
print("\n=== Test 5: Raw System Prompt ===")
raw_pred = ai.Predict(system="You are a helpful assistant. Be very concise.")
try:
    result = raw_pred(input="What is Python?")
    print(f"Result: {result[:80]}...")
    print("✓ Test 5 passed")
except Exception as e:
    print(f"❌ Test 5 failed: {e}")

# Test 6: Module Pipeline
print("\n=== Test 6: Module Pipeline ===")
class SimpleRAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict(
            "query -> classification",
            system="Classify as SQL or VECTOR"
        )
        self.answer = ai.Predict(
            "context, query -> answer",
            system="Answer based on context"
        )

    def forward(self, query):
        # Step 1: Classify
        classification = self.classify(query=query)
        print(f"  [Classify] {classification}")

        # Step 2: Mock retrieval
        context = "Sample data" if "data" in query.lower() else "Sample info"

        # Step 3: Answer
        answer = self.answer(context=context, query=query)
        print(f"  [Answer] {answer}")

        return answer

try:
    pipeline = SimpleRAG()
    answer = pipeline(query="Show me the data")
    print(f"Final: {answer}")
    print("✓ Test 6 passed")
except Exception as e:
    print(f"❌ Test 6 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("✅ All production tests completed!")
print("="*50)
