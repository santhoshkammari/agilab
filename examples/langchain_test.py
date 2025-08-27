import json

from flowgen.utils.util import get_markdown
from flowgen.tools.content_extract import web_fetch,web_search
from flowgen.tools.markdown import markdown_analyzer_get_overview,markdown_analyzer_get_header_by_line,markdown_analyzer_get_headers
from flowgen import vLLM,Agent


vllm = vLLM(
    base_url="http://192.168.170.76:8077/v1",
    model="/home/ng6309/datascience/santhosh/models/Qwen3-14B"
)

agent = Agent(llm=vllm,
              tools=[web_search,web_fetch,markdown_analyzer_get_headers,markdown_analyzer_get_header_by_line,markdown_analyzer_get_overview],
              enable_rich_debug=True)

prompt = ('what is there in langchain ollama documentation and'
          ' give me installation command mentioned in document,'
          ' by going thourhg offiical documentation step by step" , /no_think')

prompt = "get the abstract in heirarchical autoregressive transformer paper, do step by step  /no_think"
prompt = "get the abstract in attention is all you need  paper, read the official paper, do step by step  /no_think"

resp = agent(prompt)

print(resp)