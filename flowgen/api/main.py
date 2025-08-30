from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from transformers import AutoTokenizer
import json
from typing import Optional

app = FastAPI()

llm = None
tokenizer = None

class ChatRequest(BaseModel):
    messages: list
    tools: Optional[list] = None

class LoadRequest(BaseModel):
    model_path: str
    tokenizer_name: str

def get_llama():
    global llm
    if llm is None:
        raise Exception("Model not loaded. Call /load first.")
    return llm

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        raise Exception("Tokenizer not loaded. Call /load first.")
    return tokenizer

@app.post("/load")
async def load_model(request: LoadRequest):
    global llm, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_name)
        llm = Llama(
            model_path=request.model_path,
            n_ctx=1024,
            n_batch=128,
            verbose=False
        )
        return {"status": "success", "message": "Model and tokenizer loaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/unload")
async def unload_model():
    global llm, tokenizer
    llm = None
    tokenizer = None
    return {"status": "success", "message": "Model and tokenizer unloaded"}

@app.post("/chat")
async def chat(request: ChatRequest):
    model = get_llama()
    tok = get_tokenizer()
    
    def generate():
        try:
            # Apply chat template with dynamic tools
            prompt = tok.apply_chat_template(
                request.messages,
                tools=request.tools,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Reset model state before each completion
            model.reset()
            
            response = model.create_completion(
                prompt=prompt,
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