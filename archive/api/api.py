from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama, LlamaGrammar
from transformers import AutoTokenizer
import json
import re
from typing import Optional, List, Dict, Any, Union

app = FastAPI()

llm = None
tokenizer = None

class LoadOptions(BaseModel):
    n_gpu_layers: int = 0
    split_mode: int = 1
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None
    seed: int = 4294967295
    n_ctx: int = 2048
    n_batch: int = 256
    n_ubatch: int = 512
    n_threads: Optional[int] = None
    n_threads_batch: Optional[int] = None
    rope_scaling_type: int = -1
    pooling_type: int = -1
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_orig_ctx: int = 0
    logits_all: bool = False
    embedding: bool = False
    offload_kqv: bool = True
    flash_attn: bool = False
    op_offload: Optional[bool] = None
    swa_full: Optional[bool] = None
    no_perf: bool = False
    last_n_tokens_size: int = 64
    lora_base: Optional[str] = Field(None, description="Base model for LoRA adapter (leave empty if not using LoRA)")
    lora_scale: float = 1.0
    lora_path: Optional[str] = Field(None, description="Path to LoRA adapter file (leave empty if not using LoRA)")
    numa: Union[bool, int] = False
    chat_format: Optional[str] = Field(None, description="Chat template format (leave empty for auto-detection)")
    verbose: bool = False
    type_k: Optional[int] = None
    type_v: Optional[int] = None
    spm_infill: bool = False

class ChatOptions(BaseModel):
    suffix: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Union[str, List[str]] = []
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
    logit_bias: Optional[Dict[int, float]] = None

class ChatRequest(BaseModel):
    messages: list
    tools: Optional[list] = None
    options: Optional[ChatOptions] = None
    response_format: Optional[dict] = None

class LoadRequest(BaseModel):
    model_path: str
    tokenizer_name: str
    options: Optional[LoadOptions] = None

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class Choice(BaseModel):
    message: Dict[str, Any]
    finish_reason: str = "stop"

class ChatCompletion(BaseModel):
    choices: List[Choice]
    tool_calls: List[ToolCall] = []

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

def parse_tool_calls(response_text: str) -> List[ToolCall]:
    """Parse tool calls from model response"""
    tool_calls = []
    
    # Format 1: [function_name(arg=value), function_name(arg=value)]
    list_pattern = r'\[([^\]]+)\]'
    list_match = re.search(list_pattern, response_text)
    
    if list_match:
        calls_str = list_match.group(1)
        func_pattern = r'(\w+)\(([^)]+)\)'
        func_matches = re.findall(func_pattern, calls_str)
        
        for func_name, args_str in func_matches:
            try:
                arguments = {}
                arg_pairs = [arg.strip() for arg in args_str.split(',')]
                
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        
                        arguments[key] = value
                
                tool_calls.append(ToolCall(name=func_name, arguments=arguments))
                
            except Exception:
                pass
    
    # Format 2: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
    
    for match in matches:
        try:
            tool_call_data = json.loads(match.strip())
            tool_calls.append(ToolCall(
                name=tool_call_data["name"],
                arguments=tool_call_data["arguments"]
            ))
        except Exception:
            pass
    
    return tool_calls

@app.post("/load")
async def load_model(request: LoadRequest):
    global llm, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_name)
        
        # Use LoadOptions with defaults
        load_options = request.options or LoadOptions()
        load_options_dict = load_options.model_dump(exclude_none=True)
        load_options_dict["model_path"] = request.model_path
        
        # Comprehensive parameter validation
        # Remove invalid tensor_split values
        if "tensor_split" in load_options_dict and load_options_dict["tensor_split"]:
            tensor_split = load_options_dict["tensor_split"]
            if isinstance(tensor_split, list):
                # Remove if contains zeros, single element, or doesn't sum close to 1.0
                if (any(x <= 0 for x in tensor_split) or 
                    len(tensor_split) == 1 or 
                    (len(tensor_split) > 1 and abs(sum(tensor_split) - 1.0) > 0.01)):
                    del load_options_dict["tensor_split"]
        
        # Remove zero thread counts (causes crashes)
        if "n_threads" in load_options_dict and load_options_dict["n_threads"] == 0:
            del load_options_dict["n_threads"]
        
        if "n_threads_batch" in load_options_dict and load_options_dict["n_threads_batch"] == 0:
            del load_options_dict["n_threads_batch"]
        
        # Remove invalid rope_scaling_type (must be -1 or valid enum value)
        if "rope_scaling_type" in load_options_dict and load_options_dict["rope_scaling_type"] not in [-1, 0, 1, 2, 3]:
            load_options_dict["rope_scaling_type"] = -1
        
        # Remove string placeholders and empty strings that could cause crashes
        string_params = ["lora_base", "lora_path", "chat_format"]
        for param in string_params:
            if param in load_options_dict and (
                load_options_dict[param] == "string" or 
                load_options_dict[param] == "" or
                load_options_dict[param] == "null"
            ):
                del load_options_dict[param]
        
        # Validate context and batch sizes
        if "n_ctx" in load_options_dict and load_options_dict["n_ctx"] <= 0:
            load_options_dict["n_ctx"] = 2048
        
        if "n_batch" in load_options_dict and load_options_dict["n_batch"] <= 0:
            load_options_dict["n_batch"] = 256
        
        if "n_ubatch" in load_options_dict and load_options_dict["n_ubatch"] <= 0:
            load_options_dict["n_ubatch"] = 512
        
        llm = Llama(**load_options_dict)
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
    
    try:
        prompt = tok.apply_chat_template(
            request.messages,
            tools=request.tools,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model.reset()
        
        # Use ChatOptions with defaults
        chat_options = request.options or ChatOptions()
        completion_options = chat_options.model_dump(exclude_none=True)
        completion_options["prompt"] = prompt
        
        # Validate completion parameters
        if "max_tokens" in completion_options and completion_options["max_tokens"] <= 0:
            completion_options["max_tokens"] = 16
        
        if "temperature" in completion_options and completion_options["temperature"] < 0:
            completion_options["temperature"] = 0.8
        
        if "top_p" in completion_options and (completion_options["top_p"] <= 0 or completion_options["top_p"] > 1):
            completion_options["top_p"] = 0.95
        
        if "top_k" in completion_options and completion_options["top_k"] <= 0:
            completion_options["top_k"] = 40
        
        # Remove invalid model parameter (placeholder strings)
        if "model" in completion_options and (
            completion_options["model"] == "" or 
            completion_options["model"] == "string" or
            completion_options["model"] == "null"
        ):
            del completion_options["model"]
        
        # Handle JSON schema
        if request.response_format and request.response_format.get("type") == "json_object":
            schema = request.response_format.get("schema")
            if schema:
                schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
                grammar = LlamaGrammar.from_json_schema(schema_str)
                completion_options["grammar"] = grammar
        
        # Check if streaming
        if completion_options.get("stream", False):
            def generate():
                full_content = ""
                try:
                    response = model.create_completion(**completion_options)
                    for chunk in response:
                        if chunk["choices"][0]["text"]:
                            content = chunk["choices"][0]["text"]
                            full_content += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    
                    # Send final completion
                    tool_calls = parse_tool_calls(full_content)
                    final_completion = ChatCompletion(
                        choices=[Choice(
                            message={"role": "assistant", "content": full_content},
                            finish_reason="stop"
                        )],
                        tool_calls=tool_calls
                    )
                    yield f"data: {json.dumps({'completion': final_completion.model_dump()})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        
        else:
            # Non-streaming
            response = model.create_completion(**completion_options)
            content = response["choices"][0]["text"]
            tool_calls = parse_tool_calls(content)
            
            chat_completion = ChatCompletion(
                choices=[Choice(
                    message={"role": "assistant", "content": content},
                    finish_reason=response["choices"][0]["finish_reason"]
                )],
                tool_calls=tool_calls
            )
            
            return chat_completion.model_dump()
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
