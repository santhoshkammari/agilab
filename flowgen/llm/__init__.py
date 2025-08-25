# Eager import for base class (lightweight)
from .basellm import BaseLLM

# Lazy imports for heavy LLM implementations
def __getattr__(name):
    if name == "vLLM":
        from .vllm import vLLM
        return vLLM
    elif name == "hfLLM":
        from .hfllm import hfLLM
        return hfLLM
    elif name == "Ollama":
        from .ollama import Ollama
        return Ollama
    elif name == "Gemini":
        from .gemini import Gemini
        return Gemini
    elif name == "LlamaCpp":
        from .llamacpp import LlamaCpp
        return LlamaCpp
    elif name == "LFM":
        from .lfm import LFM
        return LFM
    else:
        raise AttributeError(f"module 'flowgen.llm' has no attribute '{name}'")

