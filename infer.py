from openai import OpenAI
import json
from sum import sum_numbers
from subtract import subtract_numbers

def call_function(function_name, arguments):
    if function_name == "sum_numbers":
        return sum_numbers(arguments["a"], arguments["b"])
    elif function_name == "subtract_numbers":
        return subtract_numbers(arguments["a"], arguments["b"])
    else:
        return "Unknown function"

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "sum_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract_numbers",
            "description": "Subtract second number from first number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

completion = client.chat.completions.create(
  extra_body={},
  model="deepseek/deepseek-chat-v3-0324:free",
  messages=[
    {
      "role": "user",
      "content": "What is 243 + 35?"
    }
  ],
  tools=tools,
  tool_choice="auto"
)

message = completion.choices[0].message

if message.tool_calls:
    print("Model chose to use tools:")
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        result = call_function(function_name, arguments)
        print(f"Tool: {function_name}")
        print(f"Arguments: {arguments}")
        print(f"Result: {result}")
else:
    print("Model responded directly:")
    print(message.content)