import requests
import json
from pydantic import BaseModel

class TeamInfo(BaseModel):
    team_name: str
    score: int
    city: str

def test_case_load():
    """Test the /load endpoint with options"""
    url = "http://localhost:8001/load"
    payload = {
        "model_path": "/home/ntlpt59/Downloads/LFM2-350M-F16.gguf",
        "tokenizer_name": "LiquidAI/LFM2-350M",
        "options": {
            "n_ctx": 2048,
            "n_batch": 256,
            "verbose": False
        }
    }
    response = requests.post(url, json=payload)
    print(f"Load response: {response.json()}")
    return response.status_code == 200

def test_case_chat():
    """Test the /chat endpoint with new ChatCompletion format"""
    url = "http://localhost:8001/chat"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What's 15 + 25? Also what's the weather in Paris?"}
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
            "temperature": 0.3,
            "stream": False
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("✓ Chat successful")
        print(f"Content: {result['choices'][0]['message']['content']}")
        print(f"Tool calls: {len(result['tool_calls'])}")
        for i, tool_call in enumerate(result['tool_calls']):
            print(f"  {i+1}. {tool_call['name']}({tool_call['arguments']})")
    else:
        print(f"✗ Chat failed: {response.status_code}")

def test_case_unload():
    """Test the /unload endpoint"""
    url = "http://localhost:8001/unload"
    response = requests.post(url)
    print(f"Unload response: {response.json()}")
    return response.status_code == 200


def test_case_json_format():
    """Test JSON schema enforcement with Pydantic"""
    url = "http://localhost:8001/chat"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond in valid JSON format."},
        {"role": "user", "content": "Tell me about a basketball team"}
    ]
    
    # Use Pydantic model_json_schema()
    schema = TeamInfo.model_json_schema()
    
    payload = {
        "messages": messages,
        "options": {
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": False
        },
        "response_format": {
            "type": "json_object",
            "schema": schema
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print("✓ JSON schema test successful")
        print(f"Response: {content}")
        
        try:
            parsed_json = json.loads(content.strip())
            validated = TeamInfo(**parsed_json)
            print(f"✓ Valid TeamInfo: {validated.model_dump()}")
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    else:
        print(f"✗ JSON schema test failed: {response.status_code}")

def test_case_streaming():
    """Test streaming response"""
    url = "http://localhost:8001/chat"
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 3"}
        ],
        "options": {
            "max_tokens": 30,
            "temperature": 0.5,
            "stream": True
        }
    }
    
    print("✓ Testing streaming...")
    print("Response: ", end="")
    
    with requests.post(url, json=payload, stream=True) as r:
        chunks = 0
        for line in r.iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data = line_str[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        
                        chunk_data = json.loads(data)
                        if 'content' in chunk_data:
                            print(chunk_data['content'], end="", flush=True)
                            chunks += 1
                        elif 'completion' in chunk_data:
                            completion = chunk_data['completion']
                            tool_calls = completion.get('tool_calls', [])
                            if tool_calls:
                                print(f"\n📞 Tool calls: {len(tool_calls)}")
                                
                except Exception as e:
                    pass
        
        print(f"\n✓ Received {chunks} chunks")

if __name__ == "__main__":
    print("=== FLOWGEN API QUICK TEST ===")
    
    print("\n1. Loading model...")
    test_case_load()
    
    print("\n2. Testing chat with tools...")
    test_case_chat()
    
    print("\n3. Testing JSON schema...")
    test_case_json_format()
    
    print("\n4. Testing streaming...")
    test_case_streaming()
    
    print("\n5. Unloading model...")
    test_case_unload()
    
    print("\n✅ Quick test completed!")
