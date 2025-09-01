def __getattr__(name):
    if name == "BaseLLM":
        from .llm import BaseLLM
        return BaseLLM
    elif name == "vLLM":
        from .llm import vLLM
        return vLLM
    elif name == "Ollama":
        from .llm import Ollama
        return Ollama
    elif name == "Gemini":
        from .llm import Gemini
        return Gemini
    elif name == "hfLLM":
        from .llm import hfLLM
        return hfLLM
    elif name == "LlamaCpp":
        from .llm import LlamaCpp
        return LlamaCpp
    elif name == "LFM":
        from .llm import LFM
        return LFM
    elif name == "OpenAI":
        from .llm.openai_compatible import OpenAI
        return OpenAI
    else:
        raise AttributeError(f"module 'flowgen' has no attribute '{name}'")
