from __future__ import annotations
import json
import uuid

from ollama import Client

from .basellm import BaseLLM, Restaurant, FriendList, get_weather, get_current_time, add_two_numbers


class Ollama(BaseLLM):
    """Ollama LLM implementation that inherits from BaseLLM."""
    
    def __init__(self, model="llama3.1", host="localhost:11434", **kwargs):
        # Pass host through kwargs to BaseLLM since it expects **kwargs
        super().__init__(model=model, host=host, **kwargs)

    def _load_llm(self):
        """Load the Ollama client with custom host if provided."""
        host = self._extra_params.get('host', "localhost:11434")
        if host and host != "localhost:11434":
            return Client(host=host)
        else:
            return Client()

    def chat(self, input, **kwargs):
        """Generate text using Ollama chat."""
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)
        
        # Get parameters
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        
        # Convert tools if provided
        if tools:
            tools = self._convert_function_to_tools(tools)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, tools, model=model, **kwargs)
        
        # Build options
        options = {}
        if timeout:
            options['timeout'] = timeout
        
        # Handle structured output
        format_param = self._get_format(kwargs)
        
        response = self.llm.chat(
            model=model,
            messages=input,
            tools=tools,
            format=format_param,
            options=options
        )

        
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Extract thinking content from <think> tags
        content = response.message.content or ""
        think, content = self._extract_thinking(content)
        result['think'] = think
        result['content'] = content
        
        # Check if there are tool calls
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                result['tool_calls'].append({
                    'id': getattr(tool_call, 'id', str(uuid.uuid4())),
                    'type': 'function',
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': json.dumps(tool_call.function.arguments) if isinstance(tool_call.function.arguments, dict) else tool_call.function.arguments
                    }
                })
        
        return result

    def _stream_chat(self, messages, tools, model, **kwargs):
        """Generate streaming text using Ollama chat."""
        options = {}
        if self._timeout:
            options['timeout'] = self._timeout
            
        format_param = self._get_format(kwargs)
        
        response_stream = self.llm.chat(
            model=model,
            messages=messages,
            tools=tools,
            format=format_param,
            options=options,
            stream=True
        )

        # Return a generator that yields individual chunks
        def stream_generator():
            full_content_parts = []
            all_tool_calls = []
            
            for chunk in response_stream:
                chunk_result = {"think": "", "content": "", "tool_calls": []}
                
                if hasattr(chunk, 'message') and chunk.message:
                    if hasattr(chunk.message, 'content') and chunk.message.content:
                        content = chunk.message.content
                        full_content_parts.append(content)
                        
                        chunk_result['content'] = content
                    
                    # Handle tool calls in streaming
                    if hasattr(chunk.message, 'tool_calls') and chunk.message.tool_calls:
                        for tool_call in chunk.message.tool_calls:
                            tool_call_data = {
                                'id': getattr(tool_call, 'id', str(uuid.uuid4())),
                                'type': 'function',
                                'function': {
                                    'name': tool_call.function.name,
                                    'arguments': json.dumps(tool_call.function.arguments) if isinstance(tool_call.function.arguments, dict) else tool_call.function.arguments
                                }
                            }
                            chunk_result['tool_calls'].append(tool_call_data)
                            all_tool_calls.append(tool_call_data)
                
                yield chunk_result
            
            # Final chunk with complete thinking extraction
            if full_content_parts:
                full_content = ''.join(full_content_parts)
                think, _ = self._extract_thinking(full_content)
                if think:
                    yield {"think": think, "content": "", "tool_calls": []}
        
        return stream_generator()