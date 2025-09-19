import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_file(path: str) -> str:
    """Reads the contents of a file and returns it as a string.

    Args:
        path: The file path to read from.

    Returns:
        The file contents as a string.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except IOError as e:
        return f"Error reading file {path}: {e}"


def write_file(path: str, content: str) -> str:
    """Writes content to a file.

    Args:
        path: The file path to write to.
        content: The content to write to the file.

    Returns:
        A success or error message.
    """
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {path}"
    except IOError as e:
        return f"Error writing to file {path}: {e}"


def execute_tool_call(tool_call_text):
    """Execute a tool call and return the result."""
    match = re.search(r'(\w+)\((.*?)\)', tool_call_text)
    if not match:
        return "Error: Invalid tool call format"

    func_name, args_str = match.groups()

    try:
        if '=' in args_str:
            kwargs = {}
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    kwargs[key] = value

            if func_name == "read_file":
                return read_file(**kwargs)
            elif func_name == "write_file":
                return write_file(**kwargs)
        else:
            args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
            if func_name == "read_file":
                return read_file(*args)
            elif func_name == "write_file":
                return write_file(*args)
    except Exception as e:
        return f"Error executing {func_name}: {e}"

    return f"Unknown function: {func_name}"


def print_clean(icon, action, details=""):
    """Print with clean formatting."""
    print(f"{icon} {action}" + (f" • {details}" if details else ""))


# Load model and tokenizer
print_clean("🚀", "Loading LFM2-350M...")
model_id = "LiquidAI/LFM2-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Main execution loop
prompt = "read main.py and summarize"
tools = [read_file, write_file]

print_clean("💬", "User", prompt)

for turn in range(10):  # Max 10 turns to prevent infinite loops
    print_clean("🤔", "Thinking...")

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    print('-----')
    print(response)
    print('-----')

    # Check for tool calls
    tool_call_pattern = r'<\|tool_call_start\|\>\[(.*?)\]<\|tool_call_end\|\>'
    tool_calls = re.findall(tool_call_pattern, response)

    if tool_calls:
        for tool_call in tool_calls:
            print_clean("🔧", "Tool call", tool_call)
            result = execute_tool_call(tool_call)
            print_clean("📄", "Result", result[:100] + "..." if len(result) > 100 else result)

            # Add tool result to conversation context
            prompt += f"\n\nTool result: {result}"
    else:
        # No more tool calls, show final response
        clean_response = response.split('<|tool_call_start|>')[0].strip()
        if clean_response:
            print_clean("✨", "Response", clean_response)
        break

print_clean("✅", "Done")


