from __future__ import annotations
import json
import os
import re
import requests
from typing import Any, Optional, List, Dict

from .basellm import BaseLLM


class LLM(BaseLLM):
    """HTTP client for FlowGen API server that inherits from BaseLLM."""

    def __init__(self, base_url: str | None = None, **kwargs):
        # priority: arg > env > default
        if base_url is None:
            base_url = os.getenv("BASE_URL", "http://0.0.0.0:8000")
        self._base_url = base_url.rstrip('/')
        super().__init__(**kwargs)

    def _load_llm(self):
        """No actual LLM to load - we're making HTTP requests."""
        return None

    def chat(self, input, **kwargs):
        """Generate text using HTTP API call to /chat endpoint."""
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
        
        # Prepare request payload
        payload = {
            "messages": input,
            "tools": tools,
            "options": {
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.8),
                "top_p": kwargs.get("top_p", 0.95),
                "stream": False
            }
        }
        
        # Add response format if provided
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema.model_json_schema()
                }
            elif isinstance(format_schema, dict):
                payload["response_format"] = {
                    "type": "json_object", 
                    "schema": format_schema
                }
        
        try:
            response = requests.post(
                f"{self._base_url}/chat",
                json=payload,
                timeout=timeout or 30
            )
            response.raise_for_status()
            
            # Convert API response to flowgen format
            api_result = response.json()
            return self._convert_api_response(api_result)
            
        except requests.exceptions.RequestException as e:
            return {"think": "", "content": f"API Error: {str(e)}", "tool_calls": []}
        except Exception as e:
            return {"think": "", "content": f"Error: {str(e)}", "tool_calls": []}

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using HTTP API."""
        payload = {
            "messages": messages,
            "tools": tools,
            "options": {
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.8),
                "top_p": kwargs.get("top_p", 0.95),
                "stream": True
            }
        }
        
        # Add response format if provided
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema.model_json_schema()
                }
            elif isinstance(format_schema, dict):
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema
                }
        
        def stream_generator():
            try:
                response = requests.post(
                    f"{self._base_url}/chat",
                    json=payload,
                    stream=True,
                    timeout=self._timeout or 30
                )
                response.raise_for_status()
                
                content_parts = []
                tool_calls = []
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data = line_str[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data)
                                if 'content' in chunk_data:
                                    content = chunk_data['content']
                                    if content is not None:
                                        content_parts.append(content)
                                        # Yield chunk like ollama/vllm do
                                        yield {"think": "", "content": content, "tool_calls": []}
                                elif 'completion' in chunk_data:
                                    # Handle final completion with tool calls
                                    completion = chunk_data['completion']
                                    final_result = self._convert_api_response(completion)
                                    if final_result['tool_calls']:
                                        yield {"think": "", "content": "", "tool_calls": final_result['tool_calls']}
                            except json.JSONDecodeError:
                                continue
                                
            except requests.exceptions.RequestException as e:
                yield {"think": "", "content": f"Stream Error: {str(e)}", "tool_calls": []}
        
        return stream_generator()

    def _convert_api_response(self, api_result):
        """Convert API ChatCompletion format to flowgen LLM format."""
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Handle error responses
        if "error" in api_result:
            result["content"] = f"API Error: {api_result['error']}"
            return result
        
        # Extract content from choices
        if "choices" in api_result and api_result["choices"]:
            message = api_result["choices"][0].get("message", {})
            content = message.get("content", "")
            
            # Extract thinking from content if present
            think, content = self._extract_thinking(content)
            result["think"] = think
            result["content"] = content
        
        # Extract tool calls
        if "tool_calls" in api_result and api_result["tool_calls"]:
            for tool_call in api_result["tool_calls"]:
                result["tool_calls"].append({
                    "id": f"call_{len(result['tool_calls'])}",
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"]) if isinstance(tool_call["arguments"], dict) else tool_call["arguments"]
                    }
                })
        
        return result