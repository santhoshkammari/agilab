import gradio as gr
import random
import time
from rich import print
from flowgen.llm.gemini import Gemini

llm = Gemini()

def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    response = llm(chat_history)

    chat_history.append({"role": "assistant", "content": response['content']})
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
