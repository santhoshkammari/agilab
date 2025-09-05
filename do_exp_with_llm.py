import os

from src.llm import LLM
from src.agent.agent import Agent
from pydantic import BaseModel


from src.tools.markdown import tool_functions

os.environ['BASE_URL'] = 'http://192.168.170.76:8000'

agent = Agent(system_prompt="/no_think",tools = list(tool_functions.values()))

while True:
   agent(input(">"))

while(True):
    for x in agent(input(">"),stream=True):
        print(x,flush=True)
        #if 'content' in x:
        #    print(x['content'],end="",flush=True)
