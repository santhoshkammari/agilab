from __future__ import annotations
import json
import uuid
from abc import abstractmethod, ABC
from typing import Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.base.llms.types import TextBlock, ImageBlock
from pydantic import BaseModel


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model=None, api_key=None,tools=None, format=None, timeout=None):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout
        self._api_key = api_key
        self.llm = self._load_llm()

    @abstractmethod
    def _load_llm(self):
        pass

    def chat(self, input, **kwargs):
        """Generate text using Gemini chat."""
        if self._format:
            llm = self.llm.as_structured_llm(self._format)
        else:
            llm = self.llm

        resp = llm.chat(self._normalize_input(input))

        if kwargs.get("stream") is None:
            content = resp.message.blocks[0].text
            return {
                "role": "assistant",
                "content": content,
                "model": self._model
            }

        def stream_chat(input, **kwargs):
            """Generate streaming text using Gemini chat."""
            resp_stream = llm.stream_chat(self._normalize_input(input),**kwargs)
            for chunk in resp_stream:
                yield {
                    "role": "assistant",
                    "content": chunk.delta,
                    "model": self._model
                }

        return stream_chat(input=input,**kwargs)

    def __call__(self, input, **kwargs) -> dict:
        """Generate text using the LLM. Auto-batches if input is a list of strings/prompts."""
        # Auto-batch only for list of strings or list of lists
        if isinstance(input, list) and input:
            # List of strings - batch processing
            if isinstance(input[0], str):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of lists - batch processing
            elif isinstance(input[0], list):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of dicts - treat as single conversation, not batch
            elif isinstance(input[0], dict):
                return self.chat(input=input, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")

        return self.chat(input=input, **kwargs)

    def batch_call(self, inputs, max_workers=4, **kwargs):
        """Process multiple inputs in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda text: self(text, **kwargs), inputs))
        return results

    def _normalize_input(self, input):
        """Convert string input to message format."""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = input

        chat_messages = []
        for msg in messages:
            blocks = [TextBlock(text=msg['content'])]
            
            # Handle image content if present
            if 'image' in msg and msg['image']:
                images = msg['image']
                # Convert single image to list
                if not isinstance(images, list):
                    images = [images]
                
                for img in images:
                    if isinstance(img, str):
                        # Handle string as path or URL
                        if img.startswith(('http://', 'https://')):
                            blocks.append(ImageBlock(url=img))
                        else:
                            blocks.append(ImageBlock(path=img))
                    elif isinstance(img, bytes):
                        # Handle bytes data
                        blocks.append(ImageBlock(image=img))
                    elif isinstance(img, dict):
                        # Handle dict format (already structured)
                        blocks.append(ImageBlock(**img))
                    else:
                        # Try to handle as path
                        blocks.append(ImageBlock(path=str(img)))
            
            chat_messages.append(ChatMessage(
                role=msg["role"],
                blocks=blocks
            ))
        return chat_messages

    def _check_model(self, kwargs, default_model):
        """Check if model is provided, raise error if not."""
        model = kwargs.get("model") or default_model
        if model is None:
            raise ValueError("model is None")
        return model

    def _get_tools(self, kwargs):
        """Get tools from kwargs or use default tools."""
        return kwargs.pop('tools', None) or self._tools

    def _get_format(self, kwargs):
        """Get format from kwargs or use default format."""
        return kwargs.get('format', None) or self._format

    def _get_timeout(self, kwargs):
        """Get timeout from kwargs or use default timeout."""
        timeout = kwargs.get('timeout', None) or self._timeout
        if 'timeout' in kwargs:
            kwargs.pop("timeout")
        return timeout

    def _convert_function_to_tools(self, func: Optional[list[Callable]]) -> list[dict]:
        """Convert functions to OpenAI tools format using Verifiers-style conversion."""
        if not func:
            return []
        from llm import convert_func_to_oai_tool
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]

    def _extract_thinking(self, content):
        """Extract thinking content from <think> tags."""
        import re
        think = ''
        if '<think>' in content and '</think>' in content:
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                think = think_match.group(1).strip()
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return think, content

    def _convert_tool_calls_to_dict(self, tool_calls):
        """Convert tool calls from object format to dictionary format."""
        if not tool_calls:
            return []

        dict_tool_calls = []
        for tool_call in tool_calls:
            if hasattr(tool_call, 'id'):
                # OpenAI format - convert to dict
                dict_tool_call = {
                    'id': tool_call.id,
                    'type': getattr(tool_call, 'type', 'function'),
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    }
                }
            elif isinstance(tool_call, dict):
                dict_tool_call = tool_call
            else:
                dict_tool_call = {
                    'id': str(uuid.uuid4()),
                    'type': getattr(tool_call, 'type', 'function'),
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    }
                }

            dict_tool_calls.append(dict_tool_call)

        return dict_tool_calls

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage

class Gemini(BaseLLM):
    def __init__(self, model="gemini-2.0-flash", api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE", **kwargs):
        super().__init__(model=model, api_key=api_key,**kwargs)

    def _load_llm(self):
        return GoogleGenAI(model=self._model,api_key=self._api_key)

## Test
# Test with dict format
test_messages = [
    {"role": "system", "content": "You are a pirate with a colorful personality"},
    {"role": "user", "content": "Tell me a story"}
]

# Test the Gemini class




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



llm = Gemini("gemini-2.0-flash",format=Restaurant)

print(llm("generate a restaurent in city miami"))


# # Test with __call__ method (single string)
# print("\n=== Testing with single string ===")
# resp1 = llm("Tell me a joke about pirates")
# print(f"Response: {resp1}")
#
# # Test with __call__ method (list of messages)
# print("\n=== Testing with message list ===")
# resp2 = llm(test_messages)
# print(f"Response: {resp2}")
#
# # Test with chat method directly
# print("\n=== Testing chat method directly ===")
# resp3 = llm.chat(test_messages)
# print(f"Response: {resp3}")
#
# # Test streaming
# print("\n=== Testing streaming ===")
# from llama_index.core.llms import ChatMessage
#
# stream_messages = [
#     ChatMessage(role="user", content="Tell me a short joke about programming"),
# ]
#
# gemini_llm = llm.llm  # Get the underlying GoogleGenAI instance
# resp_stream = gemini_llm.stream_chat(stream_messages)
#
# print("Streaming response:")
# full_response = ""
# for r in resp_stream:
#     print(r.delta, end="")
#     full_response += r.delta
# print(f"\n\nFull streamed response: {full_response}")
#
# # Test streaming with Gemini class
# print("\n=== Testing Gemini streaming ===")
# stream_resp = llm("Tell me a very short programming joke",stream=True)
#
# print("Gemini streaming response:")
# full_gemini_response = ""
# for chunk in stream_resp:
#     print(chunk["content"], end="")
#     print('============')
#
#     full_gemini_response += chunk["content"]
# print(f"\n\nFull Gemini streamed response: {full_gemini_response}")
#
# # Test streaming via __call__ method
# print("\n=== Testing Gemini __call__ with stream=True ===")
# stream_resp2 = llm("Why do developers hate debugging?", stream=True)
#
# print("streaming response:")
# full_call_response = ""
# for chunk in stream_resp2:
#     print(chunk["content"], end="")
#     print('============')
#     full_call_response += chunk["content"]
# print(f"\n\nFull __call__ streamed response: {full_call_response}")

