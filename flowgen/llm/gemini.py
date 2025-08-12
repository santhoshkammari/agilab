from __future__ import annotations
import json
import uuid
import inspect
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from pydantic import BaseModel

from .basellm import BaseLLM, convert_func_to_oai_tool


class Gemini(BaseLLM):
    """Google Gemini implementation using google.genai client."""
    
    def __init__(self, model="gemini-2.5-flash", api_key=None, **kwargs):
        self._api_key = api_key
        super().__init__(model=model, **kwargs)
        self.llm = self._load_llm()

    def _load_llm(self):
        """Load the Gemini client."""
        if self._api_key:
            return genai.Client(api_key=self._api_key)
        else:
            return genai.Client()

    def chat(self, input, **kwargs):
        """Generate text using Gemini chat."""
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
        
        # Build config
        config_params = {}
        
        # Handle structured output
        if format_schema:
            config_params['response_mime_type'] = 'application/json'
            config_params['response_schema'] = format_schema
        
        # Handle tools
        if tools:
            function_declarations = []
            for tool in tools:
                function_declarations.append(tool)
            config_params['tools'] = [types.Tool(function_declarations=function_declarations)]
        
        config = types.GenerateContentConfig(**config_params) if config_params else None
        
        response = self.llm.models.generate_content(
            model=self._model,
            contents=input,
            config=config
        )
        
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Check if there are function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        result['tool_calls'].append({
                            'id': str(uuid.uuid4()),
                            'type': 'function',
                            'function': {
                                'name': part.function_call.name,
                                'arguments': json.dumps(part.function_call.args)
                            }
                        })
        
        result['content'] = response.text or ""
        
        return result

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using Gemini chat."""
        config_params = {}
        
        if format_schema:
            config_params['response_mime_type'] = 'application/json'
            config_params['response_schema'] = format_schema
            
        if tools:
            function_declarations = []
            for tool in tools:
                function_declarations.append(tool)
            config_params['tools'] = [types.Tool(function_declarations=function_declarations)]
            
        config = types.GenerateContentConfig(**config_params) if config_params else None
        
        response_stream = self.llm.models.generate_content_stream(
            model=self._model,
            contents=messages,
            config=config
        )
        
        for chunk in response_stream:
            chunk_result = {"think": "", "content": "", "tool_calls": []}
            
            if chunk.text:
                chunk_result["content"] = chunk.text
                print("---", end="", flush=True)  # Show streaming progress
            
            # Check for function calls in streaming
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            chunk_result["tool_calls"].append({
                                'id': str(uuid.uuid4()),
                                'type': 'function',
                                'function': {
                                    'name': part.function_call.name,
                                    'arguments': json.dumps(part.function_call.args)
                                }
                            })
            
            yield chunk_result

    def _convert_function_to_tools(self, func: Optional[List[Callable]]) -> List[dict]:
        """Convert functions to Gemini tool format."""
        if not func:
            return []
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]

    def _normalize_input(self, input):
        """Convert string input to Gemini API format."""
        if isinstance(input, str):
            return [input]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Convert chat messages to simple content strings for Gemini
                contents = []
                for msg in input:
                    if msg.get("role") == "system":
                        contents.append(f"System: {msg['content']}")
                    elif msg.get("role") in ["user", "assistant"]:
                        contents.append(msg["content"])
                return contents
            else:
                return input
        else:
            return [input]
