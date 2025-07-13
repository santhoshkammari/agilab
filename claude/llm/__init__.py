"""
We have switched all of our code from langchain to openai.types.chat.chat_completion_message_param.

For easier transition we have
"""

# Core imports - only what we need for claude
from claude.llm.base import BaseChatModel
from claude.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)
from claude.llm.messages import (
	ContentPartImageParam as ContentImage,
)
from claude.llm.messages import (
	ContentPartRefusalParam as ContentRefusal,
)
from claude.llm.messages import (
	ContentPartTextParam as ContentText,
)
from claude.llm.ollama.chat import ChatOllama
from claude.llm.openrouter.chat import ChatOpenRouter

# Google GenAI import
try:
    from claude.llm.google_genai.chat import ChatGoogleGenAI
except ImportError:
    ChatGoogleGenAI = None

# Optional imports - only import if dependencies are available
try:
    from claude.llm.anthropic.chat import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from claude.llm.google.chat import ChatGoogle
except ImportError:
    ChatGoogle = None

try:
    from claude.llm.groq.chat import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from claude.llm.openai.chat import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# Make better names for the message

__all__ = [
	# Message types -> for easier transition from langchain
	'BaseMessage',
	'UserMessage',
	'SystemMessage',
	'AssistantMessage',
	# Content parts with better names
	'ContentText',
	'ContentRefusal',
	'ContentImage',
	# Chat models - core ones
	'BaseChatModel',
	'ChatOllama',
	'ChatOpenRouter',
	'ChatGoogleGenAI',
	# Optional chat models
	'ChatAnthropic',
	'ChatGoogle',
	'ChatGroq',
	'ChatOpenAI',
]
