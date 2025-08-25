# Eager imports for lightweight modules
from .agent import Agent, AgentChain, create_agent
from .template import BaseTemplate
from .logger import UniversalLogger, get_logger

# Lazy imports for heavy LLM modules
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
    else:
        raise AttributeError(f"module 'flowgen' has no attribute '{name}'")

__all__ = [
    'UniversalLogger',
    "get_logger",
    'BaseLLM',
    'vLLM',
    'LFM',
    'Ollama',
    'Gemini',
    'hfLLM',
    'LlamaCpp',
    'Agent',
    'AgentChain',
    'create_agent',
    'BaseTemplate'
]
