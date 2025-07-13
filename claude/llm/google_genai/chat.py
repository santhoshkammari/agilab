"""
Google Generative AI chat implementation using the new google-genai package.
"""

import asyncio
from typing import List, Optional, Any, Dict

from google import genai
from google.genai import types

from claude.llm.base import BaseChatModel
from claude.llm.messages import BaseMessage, UserMessage, SystemMessage, AssistantMessage
from claude.llm.views import ChatInvokeCompletion, ChatInvokeUsage


class ChatGoogleGenAI(BaseChatModel):
    """Google Generative AI chat model implementation."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        thinking_budget: int = 0,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self._client = None
        
        # Initialize client
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            # Will use GEMINI_API_KEY environment variable
            self._client = genai.Client()
    
    @property
    def name(self) -> str:
        """Get model name."""
        return self.model
    
    def _convert_messages_to_genai(self, messages: List[BaseMessage]) -> str:
        """Convert browser-use messages to GenAI content format."""
        # Google GenAI simple API expects a single string content
        # For more complex scenarios, we'd need to use the full chat format
        
        content_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                content_parts.append(f"System: {message.content}")
            elif isinstance(message, UserMessage):
                content_parts.append(f"User: {message.content}")
            elif isinstance(message, AssistantMessage):
                content_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(content_parts)
    
    def _convert_messages_to_genai_chat(self, messages: List[BaseMessage]) -> List[types.Content]:
        """Convert browser-use messages to GenAI chat format for more complex interactions."""
        genai_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # System messages can be included as first user message with instructions
                genai_messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"Instructions: {message.content}")]
                ))
            elif isinstance(message, UserMessage):
                genai_messages.append(types.Content(
                    role="user", 
                    parts=[types.Part(text=message.content)]
                ))
            elif isinstance(message, AssistantMessage):
                genai_messages.append(types.Content(
                    role="model",
                    parts=[types.Part(text=message.content)]
                ))
        
        return genai_messages
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> ChatInvokeCompletion:
        """Async invoke the chat model."""
        try:
            # For simple cases, use the generate_content method
            if len(messages) == 1 and isinstance(messages[0], UserMessage):
                # Simple single message
                content = messages[0].content
            else:
                # Convert to simple string format for now
                content = self._convert_messages_to_genai(messages)
            
            # Configure the request
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
            )
            
            # Make the request (this is sync, so we'll run it in an executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=content,
                    config=config
                )
            )
            
            # Extract the response text
            response_text = response.text
            
            # Create proper usage object
            usage = ChatInvokeUsage(
                prompt_tokens=0,  # GenAI doesn't provide token counts in simple API
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=0,
                total_tokens=0
            )
            
            return ChatInvokeCompletion(
                completion=response_text,
                usage=usage
            )
            
        except Exception as e:
            # Handle errors and wrap in our exception format
            from claude.llm.exceptions import ModelError
            raise ModelError(str(e)) from e
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> ChatInvokeCompletion:
        """Synchronous invoke method."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ainvoke(messages, **kwargs))
        finally:
            loop.close()