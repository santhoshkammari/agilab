import sys
sys.path.insert(0, '/home/ng6309/datascience/santhosh/products/agi/agilab/src')

from beta import dspy

# Initialize the LM with vllm provider and model
lm = dspy.LM(model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B", api_base="http://localhost:8000")

# Configure globally
dspy.configure(lm=lm)

print("=== Testing dspy.LM ===")
# Simple test with a "hi" message
print("Sending 'hi' to the model...")
response = lm("hi")

# Print the response
print("\nResponse:")
print(response)

# Extract assistant's reply
if response and "choices" in response:
    content = response["choices"][0]["message"]["content"]
    print(f"\nAssistant's reply: {content}")

print("\n=== Testing dspy.Predict ===")
# Test Predict with signature
math = dspy.Predict("question -> answer")
result = math(question="What is 2+2?")
print(f"Math result: {result}")

print("\n=== Testing dspy.Agent ===")
import asyncio

# Create agent
agent = dspy.Agent(system_prompt="You are a helpful assistant.")

# Test streaming
async def test_agent():
    print("Agent streaming response:")
    async for event in agent.stream("Tell me a short joke"):
        if event.type == "content":
            print(event.content)
        elif event.type == "end":
            print(f"\nTokens used: {event.metadata.get('usage', {}).get('total_tokens')}")

asyncio.run(test_agent())

print("\n=== dspy tests complete ===")
