"""
Agent - Streamable agent with LLM and tools support
Simple, expandable design keeping aspy's philosophy
"""
import asyncio
from typing import AsyncIterator, Optional, Any
from dataclasses import dataclass
from .lm.lm import LM


@dataclass
class Event:
    """Base event type for streaming"""
    type: str
    content: Any
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Agent:
    """
    Streamable Agent - default streaming, simple API

    Usage:
        agent = Agent(system_prompt="You are helpful assistant")

        # Stream by default
        async for event in agent.stream("Hello"):
            print(event.type, event.content)

        # Or simple call (still streams internally)
        result = await agent("What is 2+2?")
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful AI assistant.",
        lm: Optional[LM] = None,
        tools: list = None
    ):
        self.system_prompt = system_prompt
        self._lm = lm
        self.tools = tools or []
        self.conversation_history = []

    def _get_lm(self):
        """Get LM from instance or global config"""
        if self._lm:
            return self._lm

        # Try to get from global config (like aspy.configure)
        from . import get_lm
        lm = get_lm()
        if not lm:
            raise ValueError("No LM configured. Use aspy.configure(lm=...) or pass lm to Agent")
        return lm

    def _build_messages(self, user_input: str) -> list:
        """Build message list for LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        return messages

    async def stream(self, user_input: str, **params) -> AsyncIterator[Event]:
        """
        Stream agent execution (default mode)

        Yields events:
        - type="message_start": Agent starting
        - type="content": Streaming content chunks
        - type="message_end": Agent finished
        - type="tool_call": Tool being called (future)
        - type="error": Error occurred
        """
        lm = self._get_lm()

        try:
            # Emit start event
            yield Event(type="message_start", content={"role": "assistant"})

            # Build messages
            messages = self._build_messages(user_input)

            # For now, make a simple call (we'll add real streaming later)
            # This is the minimal version
            # LM is already async (uses asyncio.run internally), so call directly
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: lm(messages, **params))

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Emit content event
            yield Event(type="content", content=content)

            # Save to history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": content})

            # Emit end event
            yield Event(
                type="message_end",
                content={"role": "assistant", "content": content},
                metadata={
                    "usage": response.get("usage", {}),
                    "model": response.get("model", "unknown")
                }
            )

        except Exception as e:
            yield Event(type="error", content=str(e), metadata={"error": e})

    async def __call__(self, user_input: str, **params) -> str:
        """
        Simple call interface (collects streaming result)

        Usage:
            result = await agent("Hello")
        """
        content = ""
        async for event in self.stream(user_input, **params):
            if event.type == "content":
                content += event.content
        return content

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
