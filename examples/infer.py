from flowgen import Gemini, Agent, Ollama
from flowgen.tools.web import tool_functions as wt
from flowgen.tools.markdown import tool_functions as mt
from flowgen.tools.text_editor import tool_functions as tt
from flowgen.tools.content_extract import tool_functions as ct

# llm = Gemini()
llm = Ollama(host='192.168.170.76',model="qwen3:1.7b")

agent = Agent(llm,tools=[*list(mt.values()),*list(wt.values()),*list(ct.values())])
agent("do websearch for latest ai news, after that do search for latest modi news, and then latest ind vs eng match",temperature=0.0)
# agent("provide python code part in /home/ntlpt59/master/own/flowgen/test/transformers.md")
# agent("search for ai agents and extract information and summarize it fastly")
#agent("search for Training and Finetuning Embedding Models with Sentence Transformers v3 , and get the loss function exact code, do until you findout code.")
# agent("grpo trainer code example from official documentation transformers, get there exact code ",temperature=0.0)
# agent("virat vs rohit icc stats compare, first get virats precise information from wikipedia , then rohit and finally compare")
# agent("find batting avg, runs scored,top scores in test for rohith sharam, do wiki >> tables  >> answer, go through each table till you find answer")
