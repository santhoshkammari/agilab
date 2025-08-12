import time

from flowgen import LlamaCpp

llm = LlamaCpp(model="/home/ntlpt59/Downloads/SmolLM2-135M-Instruct-f16.gguf")

st = time.perf_counter()
print(llm('hai'))
print(time.perf_counter() - st)