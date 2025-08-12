from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from llama_cpp import Llama
from pydantic import BaseModel

from .basellm import BaseLLM, convert_func_to_oai_tool


class LlamaCpp(BaseLLM):
    """Llama.cpp implementation using llama-cpp-python client."""
    
    def __init__(self, model_path=None, n_ctx=2048, n_gpu_layers=-1, chat_format="chatml", **kwargs):
        # Store llama-cpp specific parameters
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        super().__init__(model=model_path, **kwargs)

    def _load_llm(self):
        """Load the Llama.cpp model."""
        if not self._model_path:
            raise ValueError("model_path is required for LlamaCpp")
        
        try:
            return Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                chat_format=self._chat_format,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama.cpp model {self._model_path}: {e}")

    def chat(self, input, **kwargs):
        """Generate text using Llama.cpp chat completion."""
        input = self._normalize_input(input)
        
        # Get parameters
        format_schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        
        # Convert tools if provided
        if tools:
            tools = self._convert_function_to_tools(tools)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, format_schema, tools, **kwargs)
        
        # Build completion parameters
        completion_params = {
            "messages": input,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None),
        }
        
        # Handle structured output
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model - use JSON schema mode
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
            elif isinstance(format_schema, str) and format_schema.lower() == "json":
                # Simple JSON mode
                completion_params["response_format"] = {"type": "json_object"}
        
        # Handle tools/function calling
        if tools:
            completion_params["tools"] = tools
            completion_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        response = self.llm.create_chat_completion(**completion_params)
        
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Extract content
        if response["choices"] and response["choices"][0]["message"]:
            message = response["choices"][0]["message"]
            content = message.get("content", "") or ""
            
            # Extract thinking content from <think> tags
            think, content = self._extract_thinking(content)
            result['think'] = think
            result['content'] = content
            
            # Handle tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    result['tool_calls'].append({
                        'id': tool_call.get("id", str(uuid.uuid4())),
                        'type': 'function',
                        'function': {
                            'name': tool_call["function"]["name"],
                            'arguments': tool_call["function"]["arguments"]
                        }
                    })
        
        return result

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using Llama.cpp chat completion."""
        completion_params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None),
            "stream": True,
        }
        
        # Handle structured output
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
            elif isinstance(format_schema, str) and format_schema.lower() == "json":
                completion_params["response_format"] = {"type": "json_object"}
        
        # Handle tools
        if tools:
            completion_params["tools"] = tools
            completion_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        response_stream = self.llm.create_chat_completion(**completion_params)
        
        # Return a generator that yields individual chunks
        def stream_generator():
            content_parts = []
            tool_calls = []
            
            for chunk in response_stream:
                chunk_result = {"think": "", "content": "", "tool_calls": []}
                
                if chunk["choices"] and chunk["choices"][0]["delta"]:
                    delta = chunk["choices"][0]["delta"]
                    
                    # Handle content
                    if delta.get("content"):
                        content = delta["content"]
                        content_parts.append(content)
                        chunk_result["content"] = content
                    
                    # Handle tool calls
                    if delta.get("tool_calls"):
                        for tool_call in delta["tool_calls"]:
                            tool_call_data = {
                                'id': tool_call.get("id", str(uuid.uuid4())),
                                'type': 'function',
                                'function': {
                                    'name': tool_call.get("function", {}).get("name", ""),
                                    'arguments': tool_call.get("function", {}).get("arguments", "")
                                }
                            }
                            chunk_result["tool_calls"].append(tool_call_data)
                            tool_calls.append(tool_call_data)
                
                yield chunk_result
            
            # Final processing for thinking extraction
            if content_parts:
                full_content = ''.join(content_parts)
                think, _ = self._extract_thinking(full_content)
                if think:
                    yield {"think": think, "content": "", "tool_calls": []}
        
        return stream_generator()

    def _normalize_input(self, input):
        """Convert string input to message format for Llama.cpp."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Already in message format
                return input
            else:
                # List of strings, convert to user messages
                return [{"role": "user", "content": str(item)} for item in input]
        else:
            return [{"role": "user", "content": str(input)}]
