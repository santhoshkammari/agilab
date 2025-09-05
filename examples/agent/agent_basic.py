import os

from src.agent.agent import Agent
from pydantic import BaseModel

os.environ['BASE_URL'] = 'http://192.168.170.76:8000'

class TaskTitle(BaseModel):
    title: str

# Simple usage - auto-creates LLM()
title_agent = Agent("Generate a short 3-4 word chat title")
commit_agent = Agent("Generate a concise commit message for code changes")

# Use with format
title_agent("i am working on genai, where i develop a lot of rag related things", format=TaskTitle)

commit_agent("i am working on genai, where i develop a lot of rag related things", format=TaskTitle)
