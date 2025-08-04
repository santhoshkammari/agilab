from flowgen import Gemini, Agent
from flowgen.tools.web import tool_functions as wt
from flowgen.tools.markdown import tool_functions as mt
from flowgen.tools.text_editor import tool_functions as tt
from flowgen.tools.content_extract import tool_functions as ct

llm = Gemini()
agent = Agent(llm,tools=[*list(mt.values()),*list(wt.values()),*list(ct.values())])
# agent("provide python code part in /home/ntlpt59/master/own/flowgen/test/transformers.md")
# agent("search for ai agents and extract information and summarize it fastly")
agent("search for Training and Finetuning Embedding Models with Sentence Transformers v3 , and get the loss function exact code, do until you findout code.")