from flowgen import Gemini, Agent
from flowgen.tools.web import tool_functions as wt
from flowgen.tools.markdown import tool_functions as mt
from flowgen.tools.text_editor import tool_functions as tt

llm = Gemini()
agent = Agent(llm,tools=[*list(mt.values()),*list(wt.values())])
# agent("provide python code part in /home/ntlpt59/master/own/flowgen/test/transformers.md")
agent("what is weather in gurgaon sector 48?")