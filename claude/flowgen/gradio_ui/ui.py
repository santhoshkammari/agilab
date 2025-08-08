import json

import gradio as gr
import random
import time
from rich import print
from flowgen.llm.gemini import Gemini
from flowgen import Agent

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 25°C"

llm = Gemini()
agent = Agent(llm=llm,tools=[get_weather])



async def respond(message, chat_history):
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
