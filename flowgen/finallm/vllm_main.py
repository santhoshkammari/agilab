from __future__ import annotations
import json
import uuid
import inspect
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
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
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model=None, api_key=None, tools=None, format=None, timeout=None, base_url=None):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout
        self._api_key = api_key
        self._base_url = base_url
        self.llm = self._load_llm()

    @abstractmethod
    def _load_llm(self):
        pass

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
            
            # Handle reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                thinking_parts.append(delta.reasoning_content)
        
        return {
            "think": ''.join(thinking_parts),
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
        """Convert string input to OpenAI format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Already in message format
                return input
            else:
                # List of strings, convert to user messages
                return [{"role": "user", "content": str(item)} for item in input]
        else:
            return [{"role": "user", "content": str(input)}]

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


    def _normalize_input(self, input):
        """Convert string input to OpenAI format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Already in message format
                return input
            else:
                # List of strings, convert to user messages
                return [{"role": "user", "content": str(item)} for item in input]
        else:
            return [{"role": "user", "content": str(input)}]

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

    def _convert_function_to_tools(self, func: Optional[List[Callable]]) -> List[dict]:
        """Convert functions to tool format."""
        if not func:
            return []
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]


class vLLM(BaseLLM):
    def __init__(self, model=None, api_key="EMPTY", base_url="http://localhost:8000/v1", **kwargs):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    def _load_llm(self):
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
