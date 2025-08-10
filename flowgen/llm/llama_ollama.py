from llama_index.llms.ollama import Ollama
llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

response = llm.stream_complete("Who is Paul Graham?")

for r in response:
    print(r.delta, end="")



from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)


for r in resp:
    print(r.delta, end="")


from llama_index.core.bridge.pydantic import BaseModel


class Song(BaseModel):
    """A song with name and artist."""

    name: str
    artist: str

llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

sllm = llm.as_structured_llm(Song)

from llama_index.core.llms import ChatMessage

response = sllm.chat([ChatMessage(role="user", content="Name a random song!")])
print(response.message.content)


response = await sllm.achat(
    [ChatMessage(role="user", content="Name a random song!")]
)
print(response.message.content)

response_gen = sllm.stream_chat(
    [ChatMessage(role="user", content="Name a random song!")]
)
for r in response_gen:
    print(r.message.content)


from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3.2-vision",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

messages = [
    ChatMessage(
        role="user",
        blocks=[
            TextBlock(text="What type of animal is this?"),
            ImageBlock(path="ollama_image.jpg"),
        ],
    ),
]

resp = llm.chat(messages)
print(resp)






