from __future__ import annotations
import json
import uuid
from abc import abstractmethod, ABC
from typing import Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model=None, tools=None, format=None, timeout=None):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout

    @abstractmethod
    def chat(self, input, **kwargs) -> dict:
        """Generate text using the LLM. Must be implemented by subclasses."""
        pass

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
                raise ValueError(f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")
        
        return self.chat(input=input, **kwargs)
    
    def batch_call(self, inputs, max_workers=4, **kwargs):
        """Process multiple inputs in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda text: self(text, **kwargs), inputs))
        return results

    def _normalize_input(self, input):
        """Convert string input to message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        return input

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


class GeminiLLM(BaseLLM):
    """Gemini LLM implementation using LlamaIndex GoogleGenAI."""
    
    def __init__(self, model="gemini-2.0-flash", **kwargs):
        super().__init__(model=model, **kwargs)
        from llama_index.llms.google_genai import GoogleGenAI
        from llama_index.core.llms import ChatMessage
        
        self.llm = GoogleGenAI(model=self._model)
        self.ChatMessage = ChatMessage
    
    def chat(self, input, **kwargs):
        """Generate text using Gemini chat."""
        # Normalize input to message format
        messages = self._normalize_input(input)
        
        # Convert to ChatMessage objects
        chat_messages = []
        for msg in messages:
            chat_messages.append(self.ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        # Call Gemini
        resp = self.llm.chat(chat_messages)
        
        # Extract content from response
        try:
            # Access the actual text content from the first text block
            content = resp.message.blocks[0].text
            print(f"✓ Gemini response: {content[:100]}...")
        except Exception as e:
            print(f"Error accessing response content: {e}")
            content = str(resp)  # Fallback to string conversion
        
        # Return in standard format
        return {
            "content": content,
            "role": "assistant",
            "model": self._model,
            "usage": resp.raw.get('usage_metadata', {}) if hasattr(resp, 'raw') else {},
            "raw_response": resp
        }
    
    def stream_chat(self, input, **kwargs):
        """Generate streaming text using Gemini chat."""
        # Normalize input to message format
        messages = self._normalize_input(input)
        
        # Convert to ChatMessage objects
        chat_messages = []
        for msg in messages:
            chat_messages.append(self.ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        # Call Gemini streaming
        resp_stream = self.llm.stream_chat(chat_messages)
        
        print("🌊 Starting Gemini stream...")
        
        # Return the stream generator
        for chunk in resp_stream:
            yield {
                "delta": chunk.delta,
                "content": chunk.delta,
                "role": "assistant", 
                "model": self._model,
                "raw_chunk": chunk
            }
    
    def __call__(self, input, stream=False, **kwargs):
        """Generate text using the LLM with optional streaming."""
        if stream:
            return self.stream_chat(input, **kwargs)
            
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
                raise ValueError(f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")
        
        return self.chat(input=input, **kwargs)