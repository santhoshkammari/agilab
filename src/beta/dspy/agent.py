import json
import asyncio

from dataclasses import dataclass
from transformers.utils import get_json_schema

from lm import LM

@dataclass
class AssistantResponse:
    content:str

@dataclass
class ToolCall:
    id:str
    name:str
    arguments:str

async def gen(
    lm,
    history,
    tools=None
):
    tools = [get_json_schema(x) if callable(x) else x for x in tools] if tools else []
    async for x in lm.stream(messages=history,tools=tools):
        delta = x['choices'][0]['delta']

        if 'tool_calls' in delta:
            tool_call = delta['tool_calls'][0]
            if 'id' in tool_call:
                tool_call_id = tool_call['id']
                tool_call_name = tool_call['function']['name']

            if 'arguments' in tool_call['function']:    
                yield ToolCall(id=tool_call_id,name=tool_call_name,arguments=tool_call['function']['arguments'])
        elif 'content' in delta:
            yield AssistantResponse(content=delta['content'])
        else:
            raise ValueError("Unknown delta format")
        

def get_weather(city: str, unit: str = "celsius"):
    """
    Get the current weather for a city

    Args:
        city: The name of the city
        unit: Temperature unit (choices: ["celsius", "fahrenheit"])
    """
    return f"Weather in {city}: 22 degrees {unit}"


async def main():
    # llm = LM(model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B",api_base="http://localhost:8000")
    # llm = LM(model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B")
    lm = LM()
    messages = [{"role": "user", "content": "what is weather in london and canada?, do two parallel tool calls to get the weather in both cities /no_think"}]
    tools = [get_weather]

    async for x in gen(lm=lm,history=messages,tools=tools):
        print(x,flush=True)

if __name__ == "__main__":
    asyncio.run(main())
