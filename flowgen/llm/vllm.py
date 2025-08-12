from __future__ import annotations
from openai import OpenAI
from .basellm import BaseLLM


class vLLM(BaseLLM):
    def __init__(self, model=None, api_key="EMPTY", base_url="http://localhost:8000/v1", **kwargs):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def _load_llm(self):
        """Load the vLLM OpenAI-compatible client."""
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        return client

    def chat(self, input, **kwargs):
        """Generate text using vLLM chat."""
        input = self._normalize_input(input)
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
            return self._stream_chat(input, format_schema, tools, model, timeout, **kwargs)
        
        # Handle structured output
        extra_body = {}
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model
                extra_body['response_format'] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
        
        response = self.llm.chat.completions.create(
            messages=input,
            model=model,
            tools=tools,
            extra_body=extra_body if extra_body else None,
            timeout=timeout
        )

        result = {"think": "", "content": "", "tool_calls": []}
        
        # Check if there are tool calls
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                result['tool_calls'].append({
                    'id': tool_call.id,
                    'type': 'function',
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    }
                })
        
        result['content'] = response.choices[0].message.content or ""
        # Handle reasoning_content for thinking
        result['think'] = getattr(response.choices[0].message, 'reasoning_content', "") or ""
        
        return result

    def _stream_chat(self, messages, format_schema, tools, model, timeout, **kwargs):
        """Generate streaming text using vLLM chat."""
        extra_body = {}
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                extra_body['response_format'] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
            
        response_stream = self.llm.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools,
            stream=True,
            extra_body=extra_body if extra_body else None,
            timeout=timeout
        )
        
        content_parts = []
        tool_calls = []
        thinking_parts = []
        
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content_parts.append(delta.content)
            
            # Handle tool call streaming
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.id:
                        tool_calls.append({
                            'id': tool_call.id,
                            'type': 'function',
                            'function': {
                                'name': tool_call.function.name or "",
                                'arguments': tool_call.function.arguments or ""
                            }
                        })
            
            # Handle reasoning content for thinking
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                thinking_parts.append(delta.reasoning_content)
        
        return {
            "think": ''.join(thinking_parts),
            "content": ''.join(content_parts),
            "tool_calls": tool_calls
        }
