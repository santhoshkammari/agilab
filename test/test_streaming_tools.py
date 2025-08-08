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

print("Testing streaming with tools...")
print("=" * 50)

try:
    stream = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
                "role": "user",
                "content": "Calculate 243 + 35 and then subtract 50 from the result. Show your work."
            }
        ],
        tools=tools,
        tool_choice="auto",
        stream=True
    )
    
    print("Streaming response:")
    accumulated_content = ""
    tool_calls = []
    
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            
            # Handle content streaming
            if choice.delta.content:
                print(choice.delta.content, end="", flush=True)
                accumulated_content += choice.delta.content
            
            # Handle tool calls
            if choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    if tool_call.function:
                        print(f"\n[TOOL CALL] {tool_call.function.name}: {tool_call.function.arguments}")
                        tool_calls.append({
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
            
            # Check if finished
            if choice.finish_reason:
                print(f"\n\n[FINISH REASON: {choice.finish_reason}]")
                break
    
    print("\n" + "=" * 50)
    print(f"STREAMING SUPPORT: {'YES' if accumulated_content or tool_calls else 'NO'}")
    print(f"TOOL CALLS DETECTED: {len(tool_calls)}")
    
    # Execute any tool calls that were made
    if tool_calls:
        print("\nExecuting tool calls:")
        for tool_call in tool_calls:
            try:
                args = json.loads(tool_call["arguments"])
                result = call_function(tool_call["name"], args)
                print(f"  {tool_call['name']}({args}) = {result}")
            except Exception as e:
                print(f"  Error executing {tool_call['name']}: {e}")

except Exception as e:
    print(f"Error testing streaming: {e}")
    print("STREAMING SUPPORT: UNKNOWN (Error occurred)")