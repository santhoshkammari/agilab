import json

import gradio as gr
import random
import time
from rich import print

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 25°C"

class MockAgent:
    def __init__(self, tools=None):
        self.tools = tools or []
    
    def __call__(self, messages, stream=False):
        if stream:
            return self._stream_response(messages)
        else:
            return self._sync_response(messages)
    
    def _stream_response(self, messages):
        # Mock streaming response
        events = [
            {"type": "tool_start", "tool_name": "get_weather", "tool_args": {"location": "gurgaon"}},
            {"type": "tool_result", "tool_name": "get_weather", "result": "Weather in gurgaon: Sunny, 25°C"},
            {"type": "llm_response", "content": "The weather in Gurgaon is sunny with a temperature of 25°C."}
        ]
        for event in events:
            yield event
    
    def _sync_response(self, messages):
        return {"content": "Mock response from agent"}

agent = MockAgent(tools=[get_weather])



def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    yield "",chat_history
    response_stream = agent(chat_history, stream=True)
    
    for event in response_stream:
        time.sleep(1)
        print('==============')
        print(event)
        even_type = event.get('type')
        if even_type=='tool_start':
            chat_history.append(gr.ChatMessage(
                role='assistant',
                content=f"{event.get('tool_name')}({event.get('tool_args')})",
                metadata={"title":'Searching for location','status':"pending"}
            ))
            yield "", chat_history
        elif even_type=='tool_result':
            chat_history[-1]= gr.ChatMessage(
                role='assistant',
                content=event.get('result'),
                metadata={"title":'Location Search done','status':"done"}
            )
            yield "", chat_history

        elif event.get('type') == 'llm_response':
            chat_history.append(gr.ChatMessage(
                role='assistant',
                content=event['content']
            ))
            yield "", chat_history
        else:
            yield "",chat_history
            
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(value='what is weather in gurgaon')
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
