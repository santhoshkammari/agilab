import time
import gradio as gr
from openai import OpenAI
from gradio import ChatMessage

client = OpenAI(base_url="http://0.0.0.0:8077/v1", api_key="sk-hdsfjbd")

def generate(prompt, messages):
    response = client.chat.completions.create(
        model="",
        messages=messages+ [{"role":"user","content":prompt["text"]}],
        temperature=0.7,
        stream=True
    )
    
    thought_buffer=""
    response_buffer = ""
    thinking_complete=False
    buffer = [ChatMessage(role="assistant",content="",metadata={"title":"think"})]
    
    for delta in response:
        chunk = delta.choices[0].delta.content
        
        if chunk=="<think>":
            thinking_complete=False
        elif chunk=="</think>":
            thinking_complete=True
            buffer.append(ChatMessage(role="assistant",content=thought_buffer,metadata={"title":"think"}))
        elif not thinking_complete:
            thought_buffer+=chunk
            buffer[-1]=ChatMessage(role="assistant",content=thought_buffer,metadata={"title":"think"})
        else:
            response_buffer+=chunk
            buffer[-1]=ChatMessage(role="assistant",content=response_buffer)
        
        yield buffer

demo = gr.ChatInterface(
    generate,
    type='messages',
    multimodal=True,
    theme=gr.themes.Glass(),
    fill_height=True,
    fill_width=True,
    flagging_mode="manual",
    flagging_options=["Like","Spam","Inappropriate","Other"],
    save_history=True
)

if __name__=="__main__":
    demo.launch()
