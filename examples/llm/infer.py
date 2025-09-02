from src.llm import LLM
from pydantic import BaseModel

# initialize llm
llm = LLM()

# pydantic schema
class TaskTitle(BaseModel):
    title: str

# build messages
messages = [
    {"role": "system", "content": "you are a helper that generates a short 2-3 word title for the given user query."},
    {"role": "user", "content": "i am working on genai, where i develop a lot of rag related things and ai deployment integrated"},
]

# call llm with schema
response = llm(messages, format=TaskTitle)

print(response)

