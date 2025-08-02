from __future__ import annotations
import json
from openai import OpenAI
from .llm import BaseLLM


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

        result = {"think": "", "content": "", "tools": []}
        
        result['content'] = response.choices[0].message.content or ""
        
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            for t in response.choices[0].message.tool_calls:
                result['tools'].append({
                    'name': t.function.name,
                    'arguments': json.loads(t.function.arguments)
                })

        return result