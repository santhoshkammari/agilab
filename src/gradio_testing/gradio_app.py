import gradio as gr
import websockets
import json
import asyncio

async def stream_from_websocket():
    """Stream messages from WebSocket and update UI components"""
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            async for message in websocket:
                try:
                    message_data = json.loads(message)
                    msg_type = message_data.get("type")
                    content = message_data.get("content")
                    
                    if msg_type == "keepalive":
                        yield gr.update(value=content, visible=False), gr.update()
                    elif msg_type == "assistant":
                        yield gr.update(), gr.update(value=content, visible=True)
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield gr.update(), gr.update(value=f"Error connecting to WebSocket: {str(e)}", visible=True)

with gr.Blocks() as demo:
    gr.Markdown("# FastAPI Streaming Demo")
    
    # Separate markdown components
    keepalive_display = gr.Markdown(value="", visible=False)  # Hidden keepalive updates
    assistant_display = gr.Markdown(value="", visible=True)   # Visible assistant messages
    
    start_btn = gr.Button("Start Streaming")
    
    start_btn.click(
        fn=stream_from_websocket,
        outputs=[keepalive_display, assistant_display]
    )

if __name__ == "__main__":
    demo.launch()