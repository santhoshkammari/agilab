from src.agent.agent import Agent
from pydantic import BaseModel

class TaskTitle(BaseModel):
    title: str

# Simple usage - auto-creates LLM()
title_agent = Agent("Generate a short 3-4 word chat title")
commit_agent = Agent("Generate a concise commit message for code changes")

# Use with format
title_agent("i am working on genai, where i develop a lot of rag related things", format=TaskTitle)

commit_agent("i am working on genai, where i develop a lot of rag related things", format=TaskTitle)
