# import ai

# lm = ai.LM(api_base="http://192.168.170.76:8000",temperature=0.1)
# ai.configure(lm=lm)

# pred = ai.Predict("query -> answer")
# result = pred("What is the capital of France?")
# print(result)


import dspy

# Configure LM (need api_key even for local servers)
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    api_base="http://192.168.170.76:8000",
    api_key="dummy"  # Required by litellm even for local servers
)
dspy.configure(lm=lm)

# Example 1: Simple signature-based predict
print("=== Example 1: Simple Predict ===")
pred = dspy.Predict("query -> answer")
result = pred(query="What is the capital of France?")
print(f"Answer: {result.answer}\n")

# Example 2: Using History with signature fields (NOT raw chat messages)
print("=== Example 2: With History ===")
class QA(dspy.Signature):
    """Answer questions based on conversation history."""
    query: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

pred_with_history = dspy.Predict(QA)

# First call - no history
result1 = pred_with_history(query="My name is Alice")
print(f"Q: My name is Alice\nA: {result1.answer}\n")

# Build history using signature fields (query/answer)
history = dspy.History(messages=[
    {"query": "My name is Alice", "answer": result1.answer}
])

# Second call - with history
result2 = pred_with_history(query="What is my name?", history=history)
print(f"Q: What is my name?\nA: {result2.answer}\n")

# Example 3: For raw chat messages, skip History and use direct LM call
print("=== Example 3: Raw Chat Messages ===")
raw_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What did I say?"}
]

# Direct LM call bypasses DSPy signature system
response = lm(messages=raw_messages)
print(f"Response: {response.choices[0].message.content}")
