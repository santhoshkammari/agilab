from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import json
from typing import Optional

app = FastAPI()

llm = None

class ChatRequest(BaseModel):
    prompt: str

def get_llama():
    global llm
    if llm is None:
        llm = Llama(
            model_path="/home/ntlpt59/Downloads/LFM2-350M-F16.gguf",
            n_ctx=1024,
            n_batch=128,
            verbose=False
        )
    return llm

@app.post("/chat")
async def chat(request: ChatRequest):
    model = get_llama()
    
    def generate():
        try:
            # Reset model state before each completion
            model.reset()
            
            response = model.create_completion(
                prompt=request.prompt,
                max_tokens=100,
                temperature=0.7,
                stream=True
            )
            
            for chunk in response:
                if chunk["choices"][0]["text"]:
                    content = chunk["choices"][0]["text"]
                    yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)