from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from llama_cpp import Llama
from pydantic import BaseModel

from .basellm import BaseLLM, convert_func_to_oai_tool


class LlamaCpp(BaseLLM):
    """Llama.cpp implementation using llama-cpp-python client."""
    
    def __init__(self, model_path=None, n_ctx=2048, n_gpu_layers=-1, chat_format="chatml", **kwargs):
        # Store llama-cpp specific parameters
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        super().__init__(model=model_path, **kwargs)

    def _load_llm(self):
        """Load the Llama.cpp model."""
        if not self._model_path:
            raise ValueError("model_path is required for LlamaCpp")
        
        try:
            return Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                chat_format=self._chat_format,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama.cpp model {self._model_path}: {e}")

    def chat(self, input, **kwargs):
        """Generate text using Llama.cpp chat completion."""
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
        
        # Build completion parameters
        completion_params = {
            "messages": input,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None),
        }
        
        # Handle structured output
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model - use JSON schema mode
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
            elif isinstance(format_schema, str) and format_schema.lower() == "json":
                # Simple JSON mode
                completion_params["response_format"] = {"type": "json_object"}
        
        # Handle tools/function calling
        if tools:
            completion_params["tools"] = tools
            completion_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        response = self.llm.create_chat_completion(**completion_params)
        
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Extract content
        if response["choices"] and response["choices"][0]["message"]:
            message = response["choices"][0]["message"]
            content = message.get("content", "") or ""
            
            # Extract thinking content from <think> tags
            think, content = self._extract_thinking(content)
            result['think'] = think
            result['content'] = content
            
            # Handle tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    result['tool_calls'].append({
                        'id': tool_call.get("id", str(uuid.uuid4())),
                        'type': 'function',
                        'function': {
                            'name': tool_call["function"]["name"],
                            'arguments': tool_call["function"]["arguments"]
                        }
                    })
        
        return result

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using Llama.cpp chat completion."""
        completion_params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None),
            "stream": True,
        }
        
        # Handle structured output
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_schema.__name__,
                        "schema": format_schema.model_json_schema()
                    }
                }
            elif isinstance(format_schema, str) and format_schema.lower() == "json":
                completion_params["response_format"] = {"type": "json_object"}
        
        # Handle tools
        if tools:
            completion_params["tools"] = tools
            completion_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        response_stream = self.llm.create_chat_completion(**completion_params)
        
        # Return a generator that yields individual chunks
        def stream_generator():
            content_parts = []
            tool_calls = []
            
            for chunk in response_stream:
                chunk_result = {"think": "", "content": "", "tool_calls": []}
                
                if chunk["choices"] and chunk["choices"][0]["delta"]:
                    delta = chunk["choices"][0]["delta"]
                    
                    # Handle content
                    if delta.get("content"):
                        content = delta["content"]
                        content_parts.append(content)
                        chunk_result["content"] = content
                    
                    # Handle tool calls
                    if delta.get("tool_calls"):
                        for tool_call in delta["tool_calls"]:
                            tool_call_data = {
                                'id': tool_call.get("id", str(uuid.uuid4())),
                                'type': 'function',
                                'function': {
                                    'name': tool_call.get("function", {}).get("name", ""),
                                    'arguments': tool_call.get("function", {}).get("arguments", "")
                                }
                            }
                            chunk_result["tool_calls"].append(tool_call_data)
                            tool_calls.append(tool_call_data)
                
                yield chunk_result
            
            # Final processing for thinking extraction
            if content_parts:
                full_content = ''.join(content_parts)
                think, _ = self._extract_thinking(full_content)
                if think:
                    yield {"think": think, "content": "", "tool_calls": []}
        
        return stream_generator()

    def _normalize_input(self, input):
        """Convert string input to message format for Llama.cpp."""
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


if __name__ == "__main__":
    # Initialize LlamaCpp
    # NOTE: You need to download a GGUF model file first
    # Example: wget https://huggingface.co/microsoft/DialoGPT-medium-GGUF/resolve/main/DialoGPT-medium-q4_0.gguf
    try:
        llm = LlamaCpp(
            model_path="path/to/your/model.gguf",  # Replace with actual path
            n_ctx=2048,
            n_gpu_layers=-1,  # Use all GPU layers if CUDA available
            chat_format="chatml"
        )
        
        print("=== Testing basic chat ===")
        response = llm("Tell me a short joke about programming")
        print(f"Response: {response['content']}")
        
        # Test tools with Python functions
        print("\n=== Testing tools ===")
        response = llm("What's the weather like in New York?", tools=[get_weather])
        print(f"Tool response: {response}")
        if response.get('tool_calls'):
            print("Tool calls detected - would execute functions automatically")
        
        # Test math tools
        print("\n=== Testing math tools ===")
        response = llm("What is 25 plus 17?", tools=[add_two_numbers])
        print(f"Math response: {response}")
        
        # Test structured output
        print("\n=== Testing structured output ===")
        response = llm("Generate a restaurant in Miami", format=Restaurant)
        print(f"Structured response: {response['content']}")
        
        # Test friends list structured output
        print("\n=== Testing friends list ===")
        response = llm("I have two friends. Alice is 25 and available, Bob is 30 and busy", format=FriendList)
        print(f"Friends response: {response['content']}")
        
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
        
        # Test JSON mode
        print("\n=== Testing JSON mode ===")
        response = llm("Generate a JSON object with name and age", format="json")
        print(f"JSON response: {response['content']}")
        
    except Exception as e:
        print(f"Error initializing LlamaCpp: {e}")
        print("\nTo use LlamaCpp:")
        print("1. Install: pip install llama-cpp-python")
        print("2. Download a GGUF model file")
        print("3. Update the model_path in the code")
        print("4. Run this script!")
    
    print("\n=== Simple Usage Examples ===")
    print("llm('Hello')  # Basic chat")
    print("llm('Generate person', format=PersonSchema)  # Structured output") 
    print("llm('What weather?', tools=[weather_func])  # Function calling")
    print("llm(messages)  # Multi-turn chat")
    print("llm(text, stream=True)  # Streaming")
    
    print("\nLlamaCpp Features:")
    print("• Local model execution (no API required)")
    print("• GPU acceleration support")
    print("• OpenAI-compatible API")
    print("• Function calling support")
    print("• JSON schema structured output")
    print("• Multiple chat formats (chatml, llama-2, etc.)")
    print("• Streaming generation")
    print("• Custom model loading")