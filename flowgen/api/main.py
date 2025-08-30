from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama, LlamaGrammar
from transformers import AutoTokenizer
import json
from typing import Optional

app = FastAPI()

llm = None
tokenizer = None

class ChatRequest(BaseModel):
    messages: list
    tools: Optional[list] = None
    options: Optional[dict] = None
    response_format: Optional[dict] = None

class CompletionOptions(BaseModel):
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[list] = []
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0
    top_k: int = 40
    stream: bool = False
    seed: Optional[int] = None
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    model: Optional[str] = None
    logit_bias: Optional[dict] = None

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
            
            # Use options if provided, otherwise use defaults
            completion_options = {
                "prompt": prompt,
                "stream": True
            }
            
            # Apply default options
            default_options = CompletionOptions()
            for field, value in default_options.dict().items():
                if field != "stream":  # stream is already set
                    completion_options[field] = value
            
            # Override with user-provided options
            if request.options:
                completion_options.update(request.options)
            
            # Handle response_format for JSON schema
            if request.response_format and request.response_format.get("type") == "json_object":
                schema = request.response_format.get("schema")
                if schema:
                    # Schema is already from Pydantic model_json_schema(), convert to string
                    schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
                    grammar = LlamaGrammar.from_json_schema(schema_str)
                    completion_options["grammar"] = grammar
            
            response = model.create_completion(**completion_options)
            
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