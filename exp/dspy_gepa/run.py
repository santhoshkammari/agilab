import dspy
lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://192.168.170.76:11434", api_key="")
dspy.configure(lm=lm)


from typing import Literal

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
res = classify(sentence="This book was super fun to read, though not the last chapter.")
print(res)


#res = lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
#lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
