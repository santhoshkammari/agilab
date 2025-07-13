"""
Compatibility layer to bridge old input.py interface with new browser-use LLM structure.
"""

import asyncio
from typing import List, Dict, Any, Optional
from claude.llm import BaseMessage, SystemMessage, UserMessage, AssistantMessage
from claude.llm.views import ChatInvokeCompletion


class LLMCompatibilityWrapper:
    """Wrapper to make browser-use LLMs compatible with the old input.py interface"""
    
    def __init__(self, llm_instance):
        self.llm = llm_instance
    
    def _convert_messages_to_browser_use(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert dict messages to browser-use BaseMessage objects"""
        browser_use_messages = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                browser_use_messages.append(SystemMessage(content=content))
            elif role == 'user':
                browser_use_messages.append(UserMessage(content=content))
            elif role == 'assistant':
                browser_use_messages.append(AssistantMessage(content=content))
        
        return browser_use_messages
    
    def _create_response_wrapper(self, completion: ChatInvokeCompletion) -> 'ResponseWrapper':
        """Create a response wrapper that mimics the old response format"""
        return ResponseWrapper(completion)
    
    def chat(self, messages: List[Dict[str, str]], model: str = None, tools: List = None, num_ctx: int = None, **kwargs) -> 'ResponseWrapper':
        """Synchronous chat method that wraps the async ainvoke"""
        browser_use_messages = self._convert_messages_to_browser_use(messages)
        
        # Run the async method in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            completion = loop.run_until_complete(self.llm.ainvoke(browser_use_messages))
            return self._create_response_wrapper(completion)
        finally:
            loop.close()


class MessageWrapper:
    """Wrapper to mimic the old message interface"""
    
    def __init__(self, content: str, tool_calls: List = None):
        self.content = content
        self.tool_calls = tool_calls or []


class ResponseWrapper:
    """Wrapper to mimic the old response interface"""
    
    def __init__(self, completion: ChatInvokeCompletion):
        self._completion = completion
        self.message = MessageWrapper(
            content=str(completion.completion),
            tool_calls=[]  # TODO: Handle tool calls when needed
        )