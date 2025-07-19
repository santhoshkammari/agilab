# pip install llama-index-llms-google-genai llama-index
#pip install llama-index-llms-vllm
#pip install llama-index-llms-ollama

import asyncio
from pathlib import Path
from typing import List

from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from datetime import datetime
from pydantic import BaseModel


async def example(llm):

    # conversation format
    # messages = [
    #     ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    #     ChatMessage(role="user", content="Tell me a story")
    # ]
    #

    # multimodal support
    bytes = Path("/home/ntlpt59/Pictures/Screenshots/Screenshot from 2025-07-18 17-29-14.png").read_bytes()
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                # You can pass image or even bytes
                # ImageBlock(path="/home/ntlpt59/Pictures/Screenshots/Screenshot from 2025-07-18 17-29-14.png"),
                ImageBlock(image=bytes),
                TextBlock(text="What is in this image?"),
            ],
        )
    ]

    resp = await llm.astream_chat(messages)
    async for r in resp:
        print(r.delta, end="")

async def structure_output(llm):
    class MenuItem(BaseModel):
        """A menu item in a restaurant."""

        course_name: str
        is_vegetarian: bool

    class Restaurant(BaseModel):
        """A restaurant with name, city, and cuisine."""

        name: str
        city: str
        cuisine: str
        menu_items: List[MenuItem]

    llm = llm.as_structured_llm(Restaurant)

    # conversation format
    messages = [
        ChatMessage(role="user", content="naman biryani hyderabad ,naman tomato curreny")
    ]
    resp = await llm.astream_chat(messages)
    async for r in resp:
        print(r.message.blocks, end="")
        # Output
        # [TextBlock(block_type='text', text='{"name":"naman biryani","city":"hyderabad","cuisine":"biryani","menu_items":[{"course_name":"tomato curry","is_vegetarian":true}]}')]


async def example_tool_calling(llm):
    def get_current_time(timezone: str) -> dict:
        """Get the current time"""
        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone,
        }

    def add(a:int,b:int):
        "Sum two integers"
        return a+b

    def sub(a:int,b:int):
        "Subtract two integers "
        return a-b

    first_chat_history = [
        ChatMessage(role="user", content="What is the current time in New York?")
    ]


    # uses the tool name, any type annotations, and docstring to describe the tool
    time_tool = FunctionTool.from_defaults(fn=get_current_time)
    resp = await llm.astream_chat_with_tools(tools=[time_tool], chat_history=first_chat_history)
    async for r in resp:
        print(r.model_dump()) # {'message': {'role': <MessageRole.ASSISTANT: 'assistant'>, 'additional_kwargs': {'tool_calls': [{'id': None, 'args': {'timezone': 'New York'}, 'name': 'get_current_time'}]}, 'blocks': [{'block_type': 'text', 'text': ''}]}, 'raw': {'content': {'parts': [{'video_metadata': None, 'thought': None, 'inline_data': None, 'file_data': None, 'thought_signature': None, 'code_execution_result': None, 'executable_code': None, 'function_call': {'id': None, 'args': {'timezone': 'New York'}, 'name': 'get_current_time'}, 'function_response': None, 'text': None}], 'role': 'model'}, 'citation_metadata': None, 'finish_message': None, 'token_count': None, 'finish_reason': <FinishReason.STOP: 'STOP'>, 'url_context_metadata': None, 'avg_logprobs': None, 'grounding_metadata': None, 'index': 0, 'logprobs_result': None, 'safety_ratings': None, 'usage_metadata': {'cache_tokens_details': None, 'cached_content_token_count': None, 'candidates_token_count': 18, 'candidates_tokens_details': None, 'prompt_token_count': 48, 'prompt_tokens_details': [{'modality': <MediaModality.TEXT: 'TEXT'>, 'token_count': 48}], 'thoughts_token_count': None, 'tool_use_prompt_token_count': None, 'tool_use_prompt_tokens_details': None, 'total_token_count': 66, 'traffic_type': None}}, 'delta': None, 'logprobs': None, 'additional_kwargs': {'tool_calls': [{'id': None, 'args': {'timezone': 'New York'}, 'name': 'get_current_time'}]}}

    second_chat_history = [
        ChatMessage(role="user",
                    content="2+3 and 4-5 how much , do parallel tool calling" # no need ot mention do parallel tool calling just for exmaple i added
                    )
    ]

    FunctionTool.from_defaults(fn=get_current_time)
    function_name_tools = [add,sub]
    tools = list(map(FunctionTool.from_defaults,function_name_tools))

    resp = await llm.astream_chat_with_tools(tools=tools, chat_history=second_chat_history)
    async for r in resp:
        print(r.model_dump())  # {'message': {'role': <MessageRole.ASSISTANT: 'assistant'>, 'additional_kwargs': {'tool_calls': [{'id': None, 'args': {'a': 2, 'b': 3}, 'name': 'add'}, {'id': None, 'args': {'a': 4, 'b': 5}, 'name': 'sub'}]}, 'blocks': [{'block_type': 'text', 'text': ''}]}, 'raw': {'content': {'parts': [{'video_metadata': None, 'thought': None, 'inline_data': None, 'file_data': None, 'thought_signature': None, 'code_execution_result': None, 'executable_code': None, 'function_call': {'id': None, 'args': {'a': 2, 'b': 3}, 'name': 'add'}, 'function_response': None, 'text': None}, {'video_metadata': None, 'thought': None, 'inline_data': None, 'file_data': None, 'thought_signature': None, 'code_execution_result': None, 'executable_code': None, 'function_call': {'id': None, 'args': {'a': 4, 'b': 5}, 'name': 'sub'}, 'function_response': None, 'text': None}], 'role': 'model'}, 'citation_metadata': None, 'finish_message': None, 'token_count': None, 'finish_reason': <FinishReason.STOP: 'STOP'>, 'url_context_metadata': None, 'avg_logprobs': None, 'grounding_metadata': None, 'index': 0, 'logprobs_result': None, 'safety_ratings': None, 'usage_metadata': {'cache_tokens_details': None, 'cached_content_token_count': None, 'candidates_token_count': 36, 'candidates_tokens_details': None, 'prompt_token_count': 105, 'prompt_tokens_details': [{'modality': <MediaModality.TEXT: 'TEXT'>, 'token_count': 105}], 'thoughts_token_count': None, 'tool_use_prompt_token_count': None, 'tool_use_prompt_tokens_details': None, 'total_token_count': 141, 'traffic_type': None}}, 'delta': None, 'logprobs': None, 'additional_kwargs': {'tool_calls': [{'id': None, 'args': {'a': 2, 'b': 3}, 'name': 'add'}, {'id': None, 'args': {'a': 4, 'b': 5}, 'name': 'sub'}]}}



if __name__ == '__main__':
    from llama_index.llms.ollama import Ollama
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.llms.vllm import VllmServer

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite-preview-06-17",
        api_key='AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE'
    )
    llm = Ollama(
        base_url="http://localhost:11434",
        temperature=0.0,
        model="llama3.1:latest",
        thinking=False, # thinking flag for reasonign models.
        request_timeout=120.0,
        context_window=8000# Manually set the context window to limit memory usage
    )
    llm = VllmServer(
        api_url="http://localhost:8000/generate",
        max_new_tokens=100,
        temperature=0.0
    )

    asyncio.run(example(llm))
    asyncio.run(structure_output(llm))
    asyncio.run(example_tool_calling())
