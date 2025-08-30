import requests
from transformers import AutoTokenizer
import json
from pydantic import BaseModel

def get_weather(location: str) -> str:
    """Gets the weather for a given location.
    
    Args:
        location: The location to get the weather for.
        
    Returns:
        str: A description of the weather.
    """
    # In a real implementation, you would call a weather API here.
    # For this example, we'll just return a placeholder.
    return f"The weather in {location} is sunny."

def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two integers.
    
    Args:
        a(int): The first integer.
        b: The second integer.
        
    Returns:
        int: The sum of a and b.
    """
    return a + b

model_id = "LiquidAI/LFM2-350M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Define messages
messages = [
    {"role": "system", "content": "You are a helpful assistant trained by Liquid AI."},
    {"role": "user", "content": "tell me a big joke"}
]

# 2. Define tools as JSON schema
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculates the sum of two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first integer."},
                    "b": {"type": "integer", "description": "The second integer."}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_weather",
            "description": "Gets the weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get the weather for."}
                },
                "required": ["location"]
            }
        }
    }
]

# 3. Convert messages → prompt (ChatML style)
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools_schema,
    tokenize=False,            # gives text, not tensors
    add_generation_prompt=True
)

print('=========')
print(prompt)
print('=========')

# 3. POST request with streaming enabled
url = "http://localhost:8001/chat"
headers = {"Content-Type": "application/json"}
payload = {"prompt": prompt}

with requests.post(url, json=payload, headers=headers, stream=True) as r:
    for line in r.iter_lines():
        if line:
            try:
                # Each line is like: data: {"content": "5"}
                line_str = line.decode("utf-8")
                if line_str.startswith("data:"):
                    data = line_str[len("data:"):].strip()
                    if data == "[DONE]":  # some servers send this
                        break
                    token_obj = eval(data)  # {"content": "..."}
                    print(token_obj["content"], end="", flush=True)
            except Exception as e:
                pass

print()

def test_case_load():
    """Test the /load endpoint"""
    url = "http://localhost:8001/load"
    payload = {
        "model_path": "/home/ntlpt59/Downloads/LFM2-350M-F16.gguf",
        "tokenizer_name": "LiquidAI/LFM2-350M"
    }
    response = requests.post(url, json=payload)
    print(f"Load response: {response.json()}")
    return response.status_code == 200

def test_case_chat():
    """Test the /chat endpoint with messages and tools"""
    url = "http://localhost:8001/chat"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant trained by Liquid AI."},
        {"role": "user", "content": "tell me a big joke"}
    ]
    
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Calculates the sum of two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "The first integer."},
                        "b": {"type": "integer", "description": "The second integer."}
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Gets the weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the weather for."}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    payload = {
        "messages": messages,
        "tools": tools_schema,
        "options": {
            "max_tokens": 150,
            "temperature": 0.8,
            "top_p": 0.9
        },
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"]
            }
        }
    }
    
    with requests.post(url, json=payload, stream=True) as r:
        print("Chat response:")
        for line in r.iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data = line_str[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        token_obj = json.loads(data)
                        print(token_obj["content"], end="", flush=True)
                except Exception as e:
                    pass
        print()

def test_case_unload():
    """Test the /unload endpoint"""
    url = "http://localhost:8001/unload"
    response = requests.post(url)
    print(f"Unload response: {response.json()}")
    return response.status_code == 200

class ColorResponse(BaseModel):
    color: str
    reason: str

def test_case_json_format():
    """Test the /chat endpoint with JSON format response using Pydantic schema"""
    url = "http://localhost:8001/chat"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in the specified JSON format."},
        {"role": "user", "content": "What's your favorite color and why?"}
    ]
    
    # Use Pydantic model_json_schema()
    response_schema = ColorResponse.model_json_schema()
    
    payload = {
        "messages": messages,
        "options": {
            "max_tokens": 100,
            "temperature": 0.7
        },
        "response_format": {
            "type": "json_object",
            "schema": response_schema
        }
    }
    
    print("Testing JSON format response:")
    with requests.post(url, json=payload, stream=True) as r:
        response_text = ""
        for line in r.iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data = line_str[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        token_obj = json.loads(data)
                        content = token_obj["content"]
                        response_text += content
                        print(content, end="", flush=True)
                except Exception as e:
                    pass
        print()
        
        # Try to parse the final response as JSON
        try:
            parsed_json = json.loads(response_text.strip())
            print(f"Successfully parsed JSON: {parsed_json}")
        except:
            print(f"Warning: Response was not valid JSON: {response_text}")

def test_case_tool_calling():
    """Test the /chat endpoint with tool calling"""
    url = "http://localhost:8001/chat"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the available tools when appropriate."},
        {"role": "user", "content": "What's 25 + 17? Also what's the weather in Tokyo?"}
    ]
    
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Calculates the sum of two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "The first integer."},
                        "b": {"type": "integer", "description": "The second integer."}
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Gets the weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the weather for."}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    payload = {
        "messages": messages,
        "tools": tools_schema,
        "options": {
            "max_tokens": 200,
            "temperature": 0.3
        }
    }
    
    print("Testing tool calling:")
    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data = line_str[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        token_obj = json.loads(data)
                        print(token_obj["content"], end="", flush=True)
                except Exception as e:
                    pass
        print()

if __name__ == "__main__":
    print("Testing API endpoints...")
    
    # Test load
    print("1. Testing /load endpoint")
    test_case_load()
    
    # Test regular chat
    print("2. Testing /chat endpoint")
    test_case_chat()
    
    # Test JSON format
    print("3. Testing JSON format response")
    test_case_json_format()
    
    # Test tool calling
    print("4. Testing tool calling")
    test_case_tool_calling()
    
    # Test unload
    print("5. Testing /unload endpoint") 
    test_case_unload()
