from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel

from .basellm import BaseLLM, convert_func_to_oai_tool


class vLLM(BaseLLM):
    def __init__(self, model=None, api_key="EMPTY", base_url="http://localhost:8000/v1", **kwargs):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def _load_llm(self):
        """Load the vLLM OpenAI-compatible client."""
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        
        # Get available models and set default if not specified
        if not self._model:
            try:
                models = client.models.list()
                self._model = models.data[0].id if models.data else "default"
            except Exception:
                self._model = "default"
        
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


# Test classes
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


# Test functions for tools
def get_current_time(timezone: str) -> dict:
    """Get the current time for a specific timezone"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
    }

def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    # Mock weather data - in real use case, call actual weather API
    import random
    temperatures = [15, 18, 22, 25, 28, 30, 33]
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    
    return {
        "city": city,
        "temperature": random.choice(temperatures),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80)
    }


if __name__ == "__main__":
    # Initialize vLLM
    # NOTE: Start vLLM server first with: vllm serve <model-name>
    llm = vLLM(base_url="http://localhost:8000/v1", model='HuggingFaceTB/SmolLM2-135M-Instruct')
    
    print("=== Testing basic chat ===")
    response = llm("Tell me a short joke about programming")
    print(f"Response: {response['content']}")

    # Test tools with Python functions - now automatic!
    print("\n=== Testing tools automatically ===")
    try:
        response = llm("What's the weather like in New York?", tools=[get_weather])
        print(f"Tool response: {response}")
        if response.get('tool_calls'):
            print("Tool calls detected - would execute functions automatically")
    except Exception as e:
        print(f"Error in tool calling: {e}")
        print("Make sure vLLM server is running with tool calling enabled")
    
    # Test structured output
    print("\n=== Testing structured output ===")
    try:
        response = llm("Generate a restaurant in Miami", format=Restaurant)
        print(f"Structured response: {response['content']}")
    except Exception as e:
        print(f"Error in structured output: {e}")
    
    # Test streaming
    print("\n=== Testing streaming ===")
    try:
        stream_response = llm("Tell me a short story about AI", stream=True)
        print("Streaming response:")
        if isinstance(stream_response, dict):
            print(stream_response['content'])
        else:
            for chunk in stream_response:
                print(chunk["content"], end="", flush=True)
        print()
    except Exception as e:
        print(f"Error in streaming: {e}")
    
    # Test with message history
    print("\n=== Testing message history ===")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = llm(messages)
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in message history: {e}")
    
    print("\n=== Simple Usage Examples ===")
    print("llm('Hello')  # Basic chat")
    print("llm('Generate person', format=PersonSchema)  # Structured output") 
    print("llm('What weather?', tools=[weather_func])  # Function calling")
    print("llm(messages)  # Multi-turn chat")
    print("llm(text, stream=True)  # Streaming")
    
    print("\nTo use vLLM, start the server first:")
    print("vllm serve <model-name>")
    print("Example: vllm serve microsoft/DialoGPT-medium")
    print("Or with tools: vllm serve mistralai/Mistral-7B-Instruct-v0.3 --enable-auto-tool-choice --tool-call-parser mistral")