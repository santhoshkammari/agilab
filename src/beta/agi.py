import inspect
import json
import os

import requests

def llm(messages, base_url=None, **kwargs):
    if base_url is None:
        base_url = os.getenv("BASE_URL", "http://0.0.0.0:8000")
    base_url = base_url.rstrip("/")

    payload = {
        "messages": messages,
        "tools": kwargs.get("tools"),
        "options": {
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": True,
        },
    }

    try:
        with requests.post(
            f"{base_url}/chat", json=payload, stream=True, timeout=30
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = line.decode("utf-8").removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"content": f"Error: {e}"}


def convert_func_to_oai_tool(func):
    sig = inspect.signature(func)
    props, required = {}, []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        props[name] = {"type": "string", "description": f"Parameter {name}"}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class Agent:
    def __init__(self, tools=None):
        self.tool_schemas = [convert_func_to_oai_tool(t) for t in (tools or [])]
        self.messages = []

    def __call__(self, user_content, **kwargs):
        self.messages.append({"role": "user", "content": user_content})

        response_text = []
        for chunk in llm(self.messages, tools=self.tool_schemas, **kwargs):
            if "content" in chunk and chunk["content"]:
                token = chunk["content"]
                response_text.append(token)
                yield {"type": "token", "content": token}

        if response_text:
            self.messages.append(
                {"role": "assistant", "content": "".join(response_text)}
            )

    def clear(self):
        self.messages = []


if __name__ == "__main__":
    os.environ["BASE_URL"] = "http://192.168.170.76:8000"
    agent = Agent()

    for event in agent("what is 2+3?"):
        print(event["content"], end="")

    for event in agent("What is first question  i have asked you?"):
        print(event["content"], end="")

    print("Conversation history:", agent.messages)


