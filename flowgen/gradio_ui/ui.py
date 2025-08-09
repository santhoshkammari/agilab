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
agent = Agent(llm=llm, tools=[get_weather])



def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    yield "", chat_history
    
    response_stream = agent(chat_history, stream=True)
    
    current_response = ""
    current_thinking = ""
    
    for event in response_stream:
        print(f"🎬 Event: {event}\n")
        
        # Handle different event types from agent streaming
        if event.get('type') == 'token':
            # Real-time token streaming
            current_response = event.get('accumulated_content', '')
            
            # Update chat history with current response
            updated_history = chat_history.copy()
            updated_history.append({"role": "assistant", "content": current_response})
            
            yield "", updated_history
            
        elif event.get('type') == 'tool_start':
            # Show tool being called
            tool_name = event.get('tool_name', 'unknown')
            tool_args = event.get('tool_args', {})
            
            current_response += f"\n\n🔧 Calling {tool_name}({tool_args})..."
            
            updated_history = chat_history.copy()
            updated_history.append({"role": "assistant", "content": current_response})
            
            yield "", updated_history
            
        elif event.get('type') == 'tool_result':
            # Show tool result
            tool_name = event.get('tool_name', 'unknown')
            result = event.get('result', '')
            
            current_response += f"\n✅ {tool_name}: {result}"
            
            updated_history = chat_history.copy()
            updated_history.append({"role": "assistant", "content": current_response})
            
            yield "", updated_history
            
        elif event.get('type') == 'final':
            # Final response
            final_content = event.get('content', current_response)
            
            # Update chat history with final response
            updated_history = chat_history.copy()
            updated_history.append({"role": "assistant", "content": final_content})
            
            yield "", updated_history
            break
            
        # Small delay to make streaming visible
        time.sleep(0.1)
            
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(value='what is weather in gurgaon')
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
