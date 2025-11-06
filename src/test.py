import sys
sys.path.insert(0, '/home/ng6309/datascience/santhosh/products/agi/agilab/src')

from beta import aspy

# Initialize the LM with vllm provider and model
lm = aspy.LM(model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B", api_base="http://localhost:8000")

# Simple test with a "hi" message
print("Sending 'hi' to the model...")
response = lm("hi")

# Print the response
print("\nResponse:")
print(response)

# You can also access the assistant's message content like this:
if response and "choices" in response:
    content = response["choices"][0]["message"]["content"]
    print(f"\nAssistant's reply: {content}")
