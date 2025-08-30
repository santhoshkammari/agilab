from flowgen.llm.openai_compatible import OpenAICompatible

llm = OpenAICompatible()

response = llm('what is 2+3?')

print(response)