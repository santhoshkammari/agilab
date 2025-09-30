import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import json
import re


def read_file(path: str) -> str:
    """Reads the contents of a file and returns it as a string.

    Args:
        path: The file path to read from.

    Returns:
        str: The file contents as a string.
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
        str: A success or error message.
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


print("🚀 Loading LFM2-350M...")
model_id = "LiquidAI/LFM2-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tools = [read_file, write_file]


def LFM2Chat(message, history):
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    generation_args = {
        "input_ids": input_ids,
        "do_sample": True,
        "temperature": 0.3,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
        "streamer": streamer,
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_args,
    )
    thread.start()

    acc_text = ""
    for text_token in streamer:
        time.sleep(0.1)
        acc_text += text_token

        tool_call_pattern = r'<\|tool_call_start\|\>\[(.*?)\]<\|tool_call_end\|\>'
        tool_calls = re.findall(tool_call_pattern, acc_text)

        if tool_calls:
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call)
                acc_text += f"\n\nTool result: {result}"

        clean_response = acc_text.split('<|tool_call_start|>')[0].strip()
        print(clean_response)
        yield clean_response

    thread.join()


demo = gr.ChatInterface(fn=LFM2Chat, type="messages")
demo.launch(server_name="0.0.0.0")
