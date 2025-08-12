from __future__ import annotations
import json
import uuid

from ollama import Client, chat

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
        format_schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        
        # Convert tools if provided
        if tools:
            tools = self._convert_function_to_tools(tools)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, format_schema, tools, model=model, **kwargs)
        
        # Build options
        options = {}
        if timeout:
            options['timeout'] = timeout
        
        # Handle structured output
        format_param = None
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model
                format_param = format_schema.model_json_schema()
        
        if self.llm:
            response = self.llm.chat(
                model=model,
                messages=input,
                tools=tools,
                format=format_param,
                options=options
            )
        else:
            response = chat(
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

    def _stream_chat(self, messages, format_schema, tools, model, **kwargs):
        """Generate streaming text using Ollama chat."""
        options = {}
        if self._timeout:
            options['timeout'] = self._timeout
            
        format_param = None
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                format_param = format_schema.model_json_schema()
        
        if self.llm:
            response_stream = self.llm.chat(
                model=model,
                messages=messages,
                tools=tools,
                format=format_param,
                options=options,
                stream=True
            )
        else:
            response_stream = chat(
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


if __name__ == "__main__":
    # Initialize Ollama
    # NOTE: Make sure Ollama is running with: ollama serve
    llm = Ollama(model="llama3.1", host="localhost:11434")
    
    print("=== Testing basic chat ===")
    try:
        response = llm("Tell me a short joke about programming")
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in basic chat: {e}")
        print("Make sure Ollama is running: ollama serve")
    
    # Test tools with Python functions - now automatic!
    print("\n=== Testing tools automatically ===")
    try:
        response = llm("What's the weather like in New York?", tools=[get_weather])
        print(f"Tool response: {response}")
        if response.get('tool_calls'):
            print("Tool calls detected - would execute functions automatically")
    except Exception as e:
        print(f"Error in tool calling: {e}")
        print("Make sure you have a model that supports function calling")
    
    # Test math tools
    print("\n=== Testing math tools ===")
    try:
        response = llm("What is 25 plus 17?", tools=[add_two_numbers])
        print(f"Math response: {response}")
        if response.get('tool_calls'):
            print("Math tool calls detected")
    except Exception as e:
        print(f"Error in math tools: {e}")
    
    # Test structured output
    print("\n=== Testing structured output ===")
    try:
        response = llm("Generate a restaurant in Miami", format=Restaurant)
        print(f"Structured response: {response['content']}")
    except Exception as e:
        print(f"Error in structured output: {e}")
    
    # Test friends list structured output
    print("\n=== Testing friends list ===")
    try:
        response = llm("I have two friends. Alice is 25 and available, Bob is 30 and busy", format=FriendList)
        print(f"Friends response: {response['content']}")
    except Exception as e:
        print(f"Error in friends list: {e}")
    
    # Test streaming
    print("\n=== Testing streaming ===")
    try:
        stream_response = llm("Tell me a short story about AI", stream=True)
        print("Streaming response:")
        if isinstance(stream_response, dict):
            print(stream_response['content'])
        else:
            for chunk in stream_response:
                print(chunk["content"], end="", flush=True)
        print()
    except Exception as e:
        print(f"Error in streaming: {e}")
    
    # Test with message history
    print("\n=== Testing message history ===")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = llm(messages)
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in message history: {e}")
    
    print("\n=== Simple Usage Examples ===")
    print("llm('Hello')  # Basic chat")
    print("llm('Generate person', format=PersonSchema)  # Structured output") 
    print("llm('What weather?', tools=[weather_func])  # Function calling")
    print("llm(messages)  # Multi-turn chat")
    print("llm(text, stream=True)  # Streaming")
    
    print("\nTo use Ollama:")
    print("1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
    print("2. Start: ollama serve")
    print("3. Pull model: ollama pull llama3.1")
    print("4. Run this script!")