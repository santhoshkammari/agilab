from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Callable, List

from llama_cpp import Llama

from .basellm import BaseLLM


class LlamaCpp(BaseLLM):
    """Minimal Llama.cpp wrapper that passes everything directly to llama-cpp-python."""
    
    def __init__(self, model=None, n_ctx=2048, n_gpu_layers=-1, chat_format=None, **kwargs):
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        super().__init__(model=model, **kwargs)

    def _load_llm(self):
        """Load the Llama.cpp model."""
        if not self._model:
            raise ValueError("model_path is required for LlamaCpp")
        
        try:
            return Llama(
                model_path=self._model,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                chat_format=self._chat_format,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama.cpp model {self._model}: {e}")

    def chat(self, input, **kwargs):
        """Generate text using Llama.cpp - minimal wrapper."""
        # Normalize input to messages format
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list) and all(isinstance(msg, dict) for msg in input):
            messages = input
        else:
            messages = [{"role": "user", "content": str(input)}]
        
        # Prepare parameters for llama-cpp-python create_chat_completion
        completion_params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.75),
            "top_p": kwargs.get("top_p", 0.95),
            "stop": kwargs.get("stop", None),
            "stream": kwargs.get("stream", None)
        }
        
        # Handle tools - convert functions to tool format if needed
        if "tools" in kwargs:
            tools = kwargs["tools"]
            if tools and isinstance(tools[0], (type(lambda: None), type)):
                # Convert function objects to tool schema
                from ._utils import convert_function_to_tool
                completion_params["tools"] = [convert_function_to_tool(func) for func in tools]
            else:
                completion_params["tools"] = tools
        
        # Pass through other llama-cpp-python specific parameters
        for param in ["tool_choice", "response_format", "functions", "function_call"]:
            if param in kwargs:
                completion_params[param] = kwargs[param]
        
        # Call llama-cpp-python directly

        from rich import  print
        print(completion_params)

        response = self.llm.create_chat_completion(**completion_params)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._handle_stream_response(response)
        
        # Convert to flowgen format
        result = {"think": "", "content": "", "tool_calls": []}
        
        if response["choices"] and response["choices"][0]["message"]:
            message = response["choices"][0]["message"]
            
            # Get content
            content = message.get("content") or ""
            result["content"] = content
            
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

    def _handle_stream_response(self, response_stream):
        """Handle streaming response."""
        def stream_generator():
            for chunk in response_stream:
                chunk_result = {"think": "", "content": "", "tool_calls": []}
                
                if chunk["choices"] and chunk["choices"][0]["delta"]:
                    delta = chunk["choices"][0]["delta"]
                    
                    # Handle content
                    if delta.get("content"):
                        chunk_result["content"] = delta["content"]
                    
                    # Handle tool calls
                    if delta.get("tool_calls"):
                        for tool_call in delta["tool_calls"]:
                            chunk_result["tool_calls"].append({
                                'id': tool_call.get("id", str(uuid.uuid4())),
                                'type': 'function',
                                'function': {
                                    'name': tool_call.get("function", {}).get("name", ""),
                                    'arguments': tool_call.get("function", {}).get("arguments", "")
                                }
                            })
                
                yield chunk_result
        
        return stream_generator()

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Required by BaseLLM but not used - everything goes through chat()."""
        return self.chat(messages, stream=True, **kwargs)