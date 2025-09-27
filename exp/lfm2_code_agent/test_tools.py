"""Test tools format with LFM2."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from code_tools import read_file, write_file


def get_weather(location: str) -> str:
    """Get weather information for a location.

    Args:
        location: The location to get weather for.

    Returns:
        Weather information for the location.
    """
    return f"Weather in {location}: Sunny, 72°F"


# Load model and tokenizer
print("🚀 Loading LFM2-350M...")
model_id = "LiquidAI/LFM2-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test different tool configurations
tools_configs = [
    [read_file, write_file],
    [get_weather],
    [read_file, write_file, get_weather],
]

messages = [{"role": "user", "content": "What's the weather in New York?"}]

for i, tools in enumerate(tools_configs):
    print(f"\n🧪 Testing tools config {i+1}: {[f.__name__ for f in tools]}")

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        print("✅ Tools config works!")

        # Test generation
        output = model.generate(
            input_ids.to(model.device),
            do_sample=True,
            temperature=0.3,
            max_new_tokens=100,
        )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        print(f"📝 Response: {response[:100]}...")
        break

    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Done testing tools!")