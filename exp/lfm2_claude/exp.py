from transformers import AutoTokenizer

model_id = "LiquidAI/LFM2-350M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
    {"role": "tool", "content": "Tool response here"},
    {"role": "tool", "content": "Second tool response"},
    {"role": "tool", "content": "Third tool response"},
    {"role": "tool", "content": "Fourth tool response"}
]

result = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors=None,
    tokenize=False
)

print(result)