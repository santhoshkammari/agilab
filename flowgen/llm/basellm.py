from __future__ import annotations
import json
import uuid
import inspect
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, model="", api_key=None, tools=None, format=None, timeout=None, **kwargs):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout
        self._api_key = api_key
        # Store any provider-specific parameters
        self._extra_params = kwargs
        self.llm = self._load_llm()

    @abstractmethod
    def _load_llm(self):
        """Load the LLM client/model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def chat(self, input, **kwargs):
        """Generate text using the LLM. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using the LLM. Must be implemented by subclasses."""
        pass

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
        """Convert string input to message format."""
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
        """Get format from kwargs or use default format and convert to appropriate schema."""
        format_schema = kwargs.get('format', None) or self._format
        if not format_schema:
            return None

        return format_schema

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

    def _extract_thinking(self, content):
        """Extract thinking content from <think> tags."""
        import re
        think = ''
        if '<think>' in content:
            if '</think>' in content:
                # Complete thinking block
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    think = think_match.group(1).strip()
                    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
            else:
                # Incomplete thinking block - extract everything after <think>
                think_match = re.search(r'<think>(.*)', content, re.DOTALL)
                if think_match:
                    think = think_match.group(1).strip()
                    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        return think, content


# Shared Pydantic models for testing
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


class FriendInfo(BaseModel):
    """A friend's information."""
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    """A list of friends."""
    friends: List[FriendInfo]


class Person(BaseModel):
    """A person with basic information."""
    name: str
    age: int
    is_available: bool


# Test utility functions
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


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)


if __name__ == '__main__':
    print(convert_func_to_oai_tool(add_two_numbers))
