from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

resp = llm.complete("Who is Paul Graham?")
print(resp)


from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
llm = GoogleGenAI(model="gemini-2.0-flash")
resp = llm.chat(messages)

print(resp)



from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")

resp = llm.stream_complete("Who is Paul Graham?")
for r in resp:
    print(r.delta, end="")


from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="Who is Paul Graham?"),
]

resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")



from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")

resp = await llm.astream_complete("Who is Paul Graham?")
async for r in resp:
    print(r.delta, end="")



messages = [
    ChatMessage(role="user", content="Who is Paul Graham?"),
]

resp = await llm.achat(messages)
print(resp)


from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="image.jpg"),
            TextBlock(text="What is in this image?"),
        ],
    )
]

resp = llm.chat(messages)
print(resp)


from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel
from typing import List


class MenuItem(BaseModel):
    """A menu item in a restaurant."""

    course_name: str
    is_vegetarian: bool


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


llm = GoogleGenAI(model="gemini-2.0-flash")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

# Option 1: Use `as_structured_llm`
restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw
)
# Option 2: Use `structured_predict`
# restaurant_obj = llm.structured_predict(Restaurant, prompt_tmpl, city_name="Miami")


from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from datetime import datetime

llm = GoogleGenAI(model="gemini-2.0-flash")


def get_current_time(timezone: str) -> dict:
    """Get the current time"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
    }


# uses the tool name, any type annotations, and docstring to describe the tool
tool = FunctionTool.from_defaults(fn=get_current_time)


resp = llm.predict_and_call([tool], "What is the current time in New York?")
print(resp)


chat_history = [
    ChatMessage(role="user", content="What is the current time in New York?")
]
tools_by_name = {t.metadata.name: t for t in [tool]}

resp = llm.chat_with_tools([tool], chat_history=chat_history)
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

if not tool_calls:
    print(resp)
else:
    while tool_calls:
        # add the LLM's response to the chat history
        chat_history.append(resp.message)

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs

            print(f"Calling {tool_name} with {tool_kwargs}")
            tool_output = tool.call(**tool_kwargs)
            print("Tool output: ", tool_output)
            chat_history.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_output),
                    # most LLMs like Gemini, Anthropic, OpenAI, etc. need to know the tool call id
                    additional_kwargs={"tool_call_id": tool_call.tool_id},
                )
            )

            resp = llm.chat_with_tools([tool], chat_history=chat_history)
            tool_calls = llm.get_tool_calls_from_response(
                resp, error_on_no_tool_call=False
            )
    print("Final response: ", resp.message.content)


# Define another tool for temperature
def get_temperature(city: str) -> dict:
    """Get the current temperature for a city"""
    return {
        "city": city,
        "temperature": "25°C",
    }


# Create tools from functions
tool1 = FunctionTool.from_defaults(fn=get_current_time)
tool2 = FunctionTool.from_defaults(fn=get_temperature)

# Ask a question that requires both tools
chat_history = [
    ChatMessage(
        role="user",
        content="What is the current time and temperature in New York?",
    )
]

# The model will intelligently decide which tools to call
resp = llm.chat_with_tools([tool1, tool2], chat_history=chat_history)
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

print(f"Model made {len(tool_calls)} tool calls:")
for i, tool_call in enumerate(tool_calls, 1):
    print(f"{i}. {tool_call.tool_name} with args: {tool_call.tool_kwargs}")



