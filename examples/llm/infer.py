from src.llm import LLM

llm = LLM('http://192.168.170.76:8000')

response = llm('hai')

print(response)
