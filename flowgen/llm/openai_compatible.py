from __future__ import annotations
import json
from typing import Any, Optional, List, Dict, Union
from openai import OpenAI

from .basellm import BaseLLM


class OpenAI(BaseLLM):
    """OpenAI LLM wrapper for OpenAI API-compatible services."""
    
    def __init__(self, model="", api_key="EMPTY", base_url="http://0.0.0.0:8000/v1", 
                 organization=None, project=None, **kwargs):
        """
        Initialize OpenAI-compatible client.
        
        Args:
            model: Model name to use
            api_key: API key for authentication
            base_url: Base URL for the API (defaults to OpenAI)
            organization: Organization ID for OpenAI
            project: Project ID for OpenAI
            **kwargs: Additional parameters passed to BaseLLM
        """
        self._base_url = base_url
        self._organization = organization
        self._project = project
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _load_llm(self):
        """Load the OpenAI-compatible client."""
        client_kwargs = {
            "api_key": self._api_key,
            "base_url": self._base_url,
        }
        
        # Add optional parameters if provided
        if self._organization:
            client_kwargs["organization"] = self._organization
        if self._project:
            client_kwargs["project"] = self._project
            
        return OpenAI(**client_kwargs)

    def chat(self, input, **kwargs):
        """Generate text using OpenAI-compatible chat API."""
        # Normalize input to messages format
        messages = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)

        # Get parameters
        format_schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)

        # Convert tools if provided
        if tools:
            tools = self._convert_function_to_tools(tools)

        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(messages, format_schema, tools, model, timeout, **kwargs)

        # Prepare chat completion parameters
        completion_params = {
            "messages": messages,
            "model": model,
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", None),
            "top_p": kwargs.get("top_p", None),
            "frequency_penalty": kwargs.get("frequency_penalty", None),
            "presence_penalty": kwargs.get("presence_penalty", None),
            "stop": kwargs.get("stop", None),
            "timeout": timeout,
        }

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if kwargs.get("tool_choice"):
                completion_params["tool_choice"] = kwargs["tool_choice"]

        # Handle structured output (response_format)
        if format_schema:
            if isinstance(format_schema, dict):
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": format_schema
                    }
                }
            else:
                # For string formats like "json_object"
                completion_params["response_format"] = {"type": format_schema}

        # Remove None values to avoid API errors
        completion_params = {k: v for k, v in completion_params.items() if v is not None}

        # Make API call
        response = self.llm.chat.completions.create(**completion_params)

        # Convert to flowgen format
        result = {"think": "", "content": "", "tool_calls": []}

        if response.choices and response.choices[0].message:
            message = response.choices[0].message
            
            # Get content
            result["content"] = message.content or ""
            
            # Extract thinking from content if present
            think, content = self._extract_thinking(result["content"])
            result["think"] = think
            result["content"] = content
            
            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result['tool_calls'].append({
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    })

        return result

    def _stream_chat(self, messages, format_schema, tools, model, timeout, **kwargs):
        """Generate streaming text using OpenAI-compatible chat API."""
        # Prepare streaming parameters
        completion_params = {
            "messages": messages,
            "model": model,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", None),
            "top_p": kwargs.get("top_p", None),
            "frequency_penalty": kwargs.get("frequency_penalty", None),
            "presence_penalty": kwargs.get("presence_penalty", None),
            "stop": kwargs.get("stop", None),
            "timeout": timeout,
        }

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if kwargs.get("tool_choice"):
                completion_params["tool_choice"] = kwargs["tool_choice"]

        # Handle structured output
        if format_schema:
            if isinstance(format_schema, dict):
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": format_schema
                    }
                }
            else:
                completion_params["response_format"] = {"type": format_schema}

        # Remove None values
        completion_params = {k: v for k, v in completion_params.items() if v is not None}

        # Create streaming response
        response_stream = self.llm.chat.completions.create(**completion_params)

        def stream_generator():
            accumulated_content = ""
            accumulated_thinking = ""
            tool_calls = {}
            
            for chunk in response_stream:
                chunk_result = {"think": "", "content": "", "tool_calls": []}
                
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Handle content
                    if delta.content:
                        accumulated_content += delta.content
                        # Extract thinking if present
                        think, content = self._extract_thinking(accumulated_content)
                        if think != accumulated_thinking:
                            chunk_result["think"] = think[len(accumulated_thinking):]
                            accumulated_thinking = think
                        chunk_result["content"] = delta.content
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            index = tool_call_delta.index
                            
                            if index not in tool_calls:
                                tool_calls[index] = {
                                    'id': tool_call_delta.id or "",
                                    'type': 'function',
                                    'function': {
                                        'name': "",
                                        'arguments': ""
                                    }
                                }
                            
                            # Update tool call data
                            if tool_call_delta.id:
                                tool_calls[index]['id'] = tool_call_delta.id
                            
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_calls[index]['function']['name'] += tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments
                            
                            # Add to chunk result
                            chunk_result["tool_calls"] = [tool_calls[index]]
                
                yield chunk_result
        
        return stream_generator()

    def completions(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Legacy completions endpoint for backward compatibility."""
        completion_params = {
            "model": self._check_model(kwargs, self._model),
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", None),
            "top_p": kwargs.get("top_p", None),
            "frequency_penalty": kwargs.get("frequency_penalty", None),
            "presence_penalty": kwargs.get("presence_penalty", None),
            "stop": kwargs.get("stop", None),
            "timeout": self._get_timeout(kwargs),
        }
        
        # Remove None values
        completion_params = {k: v for k, v in completion_params.items() if v is not None}
        
        response = self.llm.completions.create(**completion_params)
        
        result = {"think": "", "content": "", "tool_calls": []}
        
        if response.choices:
            content = response.choices[0].text or ""
            think, content = self._extract_thinking(content)
            result["think"] = think
            result["content"] = content
        
        return result

    def embeddings(self, input: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings using OpenAI-compatible embeddings API."""
        model = kwargs.get("model") or "text-embedding-ada-002"
        
        response = self.llm.embeddings.create(
            input=input,
            model=model,
            **{k: v for k, v in kwargs.items() if k not in ["model"]}
        )
        
        return {
            "embeddings": [data.embedding for data in response.data],
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None
        }
