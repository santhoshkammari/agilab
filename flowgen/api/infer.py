import requests
from transformers import AutoTokenizer

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
        a: The first integer.
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

# 2. Convert messages → prompt (ChatML style)
prompt = tokenizer.apply_chat_template(
    messages,
    tools=[calculate_sum,get_weather],
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
