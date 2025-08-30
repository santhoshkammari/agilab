import requests
from transformers import AutoTokenizer

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
