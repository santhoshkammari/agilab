import gradio as gr

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type your message here...", container=False, scale=7)

demo.launch(server_port=7862)