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


# Test classes for demonstration
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
    # Initialize Gemini
    llm = Gemini("gemini-2.5-flash")
    
    print("=== Testing basic chat ===")
    response = llm("Tell me a short joke about programming")
    print(f"Response: {response['content']}")
    
    # Test tools with Python functions - now automatic!
    print("\n=== Testing tools automatically ===")
    response = llm("What's the weather like in New York?", tools=[get_weather])
    print(f"Tool response: {response}")
    if response.get('tool_calls'):
        print("Tool calls detected - would execute functions automatically")
    
    # Test structured output
    print("\n=== Testing structured output ===")
    response = llm("Generate a restaurant in Miami", format=Restaurant)
    print(f"Structured response: {response['content']}")
    
    # Test streaming
    print("\n=== Testing streaming ===")
    stream_response = llm("Tell me a short story about AI", stream=True)
    print("Streaming response:")
    if isinstance(stream_response, dict):
        print(stream_response['content'])
    else:
        for chunk in stream_response:
            print(chunk["content"], end="", flush=True)
        print()
    
    # Test with message history
    print("\n=== Testing message history ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = llm(messages)
    print(f"Response: {response['content']}")
    
    # Test simple weather function call - just pass function in tools
    print("\n=== Testing simple weather function ===")
    def simple_weather(location: str) -> str:
        """Get weather for a location"""
        return f"Weather in {location}: 25°C, sunny"
    
    weather_response = llm("What's the weather in Paris?", tools=[simple_weather])
    print(f"Weather tool response: {weather_response}")
    
    print("\n=== Simple Usage Examples ===")
    print("llm('Hello')  # Basic chat")
    print("llm('Generate person', format=PersonSchema)  # Structured output")
    print("llm('What weather?', tools=[weather_func])  # Function calling")
    print("llm(messages)  # Multi-turn chat")
    print("llm(texts, stream=True)  # Streaming")