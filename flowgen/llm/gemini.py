from __future__ import annotations
import json
from openai import OpenAI
from flowgen.llm import BaseLLM


class Gemini(BaseLLM):
    def __init__(self, model="gemini-2.5-flash", tools=None, format=None, timeout=None, api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE"):
        super().__init__(model=model, tools=tools, format=format, timeout=timeout)
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def chat(self, input, **kwargs) -> dict:
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)
        tools = self._get_tools(kwargs)
        schema = self._get_format(kwargs)
        timeout = self._get_timeout(kwargs)
        
        if tools:
            tools = self._convert_function_to_tools(tools)
            # Clean up tools for Gemini compatibility
            cleaned_tools = []
            for tool in tools:
                cleaned_tool = {
                    'type': tool.get('type', 'function'),
                    'function': {
                        'name': tool['function']['name'],
                        'description': tool['function']['description'],
                        'parameters': {
                            'type': 'object',
                            'properties': {},
                            'required': tool['function']['parameters'].get('required', [])
                        }
                    }
                }
                # Clean properties
                for prop_name, prop_def in tool['function']['parameters'].get('properties', {}).items():
                    cleaned_tool['function']['parameters']['properties'][prop_name] = {
                        k: v for k, v in prop_def.items() 
                        if v is not None and k in ['type', 'description', 'enum']
                    }
                cleaned_tools.append(cleaned_tool)
            kwargs['tools'] = cleaned_tools
            
        # Remove tools from kwargs if None
        if 'tools' in kwargs and not kwargs['tools']:
            kwargs.pop('tools')
            
        # Handle structured output using JSON schema
        if schema:
            kwargs['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema
                }
            }
            # Remove format from kwargs
            kwargs.pop('format', None)

        response = self._client.chat.completions.create(
            model=model,
            messages=input,
            timeout=timeout,
            **kwargs
        )

        result = {"think": "", "content": "", "tool_calls": []}
        
        # Handle streaming response
        if hasattr(response, '__class__') and 'Stream' in str(response.__class__):
            # This is a streaming response - collect all chunks
            content_parts = []
            tool_calls = []
            
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Handle content
                    if hasattr(choice, 'delta') and choice.delta:
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            content_parts.append(choice.delta.content)
                        
                        # Handle tool calls
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            for tool_call in choice.delta.tool_calls:
                                if hasattr(tool_call, 'function'):
                                    tool_calls.append(tool_call)
            
            result['content'] = ''.join(content_parts)
            
            # Process tool calls
            if tool_calls:
                for t in tool_calls:
                    if hasattr(t, 'function') and t.function:
                        result['tools'].append({
                            'name': t.function.name,
                            'arguments': json.loads(t.function.arguments or '{}')
                        })
        else:
            # This is a regular response
            result['content'] = response.choices[0].message.content or ""
            
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                result['tool_calls']  = self._convert_tool_calls_to_dict(response.choices[0].message.tool_calls)
                # for t in response.choices[0].message.tool_calls:
                #     result['tool_calls'].append({
                #         'name': t.function.name,
                #         'arguments': json.loads(t.function.arguments)
                #     })
        return result