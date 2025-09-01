import time
from llama_cpp import Llama, LlamaDiskCache

cache = LlamaDiskCache('./lfm2_cache')

# Create ONE instance and reuse it
llm = Llama(
    model_path="/home/ntlpt59/master/models/LFM2-350M-F16.gguf",
    verbose=False,
    cache=cache,
    seed=42,
    n_ctx=2048,
)

# Test different prompts with shared system prompt
system_prompt = """You are a helpful AI assistant. Answer briefly."""

questions = [
    "What is 2+2?",
    "What is 3+3?", 
    "What is the capital of France?",
    "What is the capital of Spain?",
    "What is 2+2?",  # Repeat - should be much faster due to cache
]

print("Testing cache with same Llama instance + different prompts")
print("Run | Question | Time (s)")
print("----|----------|--------")

for i, question in enumerate(questions):
    full_prompt = system_prompt + "\n\nQ: " + question + "\nA:"
    
    st = time.perf_counter()
    try:
        output = llm(prompt=full_prompt, max_tokens=20, temperature=0.0)
        et = time.perf_counter()
        runtime = et - st
        status = "✓"
    except Exception as e:
        et = time.perf_counter()
        runtime = et - st
        status = "✗"
    
    print(f"{i+1:3} | Q{i+1:1}       | {runtime:6.2f} {status}")

print("\nNote: Repeated prompts (like Q5=Q1) should show cache speedup")