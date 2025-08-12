from __future__ import annotations
import json
import uuid
import inspect
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from pydantic import BaseModel


def convert_func_to_oai_tool(func: Any) -> dict:
    """Convert function to OpenAI function-calling tool schema."""
    if not callable(func):
        raise TypeError("Expected a callable object")

    signature = inspect.signature(func)
    hints = get_type_hints(func)

    # Map Python types to JSON schema types
    type_map = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    properties = {}
    required = []

    for name, param in signature.parameters.items():
        if name == "self":
            continue

        param_type = hints.get(name, str)
        json_type = type_map.get(param_type, "string")
        
        properties[name] = {
            "type": json_type,
            "description": f"Parameter {name}"
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": func.__doc__ or f"Call {func.__name__}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model=None, api_key=None, tools=None, format=None, timeout=None):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout
        self._api_key = api_key
        self.llm = self._load_llm()

    @abstractmethod
    def _load_llm(self):
        pass

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
        
        content_parts = []
        tool_calls = []
        
        for chunk in response_stream:
            if chunk.text:
                content_parts.append(chunk.text)
            
            # Check for function calls in streaming
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_calls.append({
                                'id': str(uuid.uuid4()),
                                'type': 'function',
                                'function': {
                                    'name': part.function_call.name,
                                    'arguments': json.dumps(part.function_call.args)
                                }
                            })
        
        return {
            "think": "",
            "content": ''.join(content_parts),
            "tool_calls": tool_calls
        }

    def __call__(self, input, **kwargs) -> dict:
        """Generate text using the LLM. Auto-batches if input is a list of strings/prompts."""
        # Auto-batch only for list of strings or list of lists
        if isinstance(input, list) and input:
            # List of strings - batch processing
            if isinstance(input[0], str):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of lists - batch processing
            elif isinstance(input[0], list):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of dicts - treat as single conversation, not batch
            elif isinstance(input[0], dict):
                return self.chat(input=input, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")

        return self.chat(input=input, **kwargs)

    def batch_call(self, inputs, max_workers=4, **kwargs):
        """Process multiple inputs in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda text: self(text, **kwargs), inputs))
        return results

    def _normalize_input(self, input):
        """Convert string input to Gemini API format."""
        if isinstance(input, str):
            return [input]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Convert chat messages to simple content strings
                contents = []
                for msg in input:
                    if msg.get("role") == "system":
                        # System messages can be handled as regular content for now
                        contents.append(f"System: {msg['content']}")
                    elif msg.get("role") in ["user", "assistant"]:
                        contents.append(msg["content"])
                return contents
            else:
                return input
        else:
            return [input]

    def _check_model(self, kwargs, default_model):
        """Check if model is provided, raise error if not."""
        model = kwargs.get("model") or default_model
        if model is None:
            raise ValueError("model is None")
        return model

    def _get_tools(self, kwargs):
        """Get tools from kwargs or use default tools."""
        return kwargs.pop('tools', None) or self._tools

    def _get_format(self, kwargs):
        """Get format from kwargs or use default format."""
        return kwargs.get('format', None) or self._format

    def _get_timeout(self, kwargs):
        """Get timeout from kwargs or use default timeout."""
        timeout = kwargs.get('timeout', None) or self._timeout
        if 'timeout' in kwargs:
            kwargs.pop("timeout")
        return timeout

    def _extract_thinking(self, content):
        """Extract thinking content from <think> tags."""
        import re
        think = ''
        if '<think>' in content and '</think>' in content:
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                think = think_match.group(1).strip()
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return think, content


class Gemini(BaseLLM):
    def __init__(self, model="gemini-2.5-flash", api_key=None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _load_llm(self):
        if self._api_key:
            return genai.Client(api_key=self._api_key)
        else:
            return genai.Client()

    def _get_tools(self, kwargs):
        """Get tools from kwargs or use default tools."""
        return kwargs.pop('tools', None) or self._tools

    def _get_format(self, kwargs):
        """Get format from kwargs or use default format."""
        return kwargs.get('format', None) or self._format

    def _get_timeout(self, kwargs):
        """Get timeout from kwargs or use default timeout."""
        timeout = kwargs.get('timeout', None) or self._timeout
        if 'timeout' in kwargs:
            kwargs.pop("timeout")
        return timeout

    def _convert_function_to_tools(self, func: Optional[List[Callable]]) -> List[dict]:
        """Convert functions to tool format."""
        if not func:
            return []
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]

    def _normalize_input(self, input):
        """Convert string input to message format."""
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
