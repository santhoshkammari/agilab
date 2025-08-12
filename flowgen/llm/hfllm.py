from __future__ import annotations
import json
import uuid
import inspect
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

import torch
import xgrammar as xgr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pydantic import BaseModel

import sys
import os
from .basellm import BaseLLM, convert_func_to_oai_tool


class hfLLM(BaseLLM):
    def __init__(self, model, device=None, torch_dtype=None, **kwargs):
        self._torch_dtype = torch_dtype or torch.float32
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model=model, **kwargs)

    def _load_llm(self):
        """Load hfLLM model and tokenizer."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._model)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load config
            config = AutoConfig.from_pretrained(self._model)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self._model,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )
            
            return model, tokenizer, config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load hfLLM model {self._model}: {e}")

    def chat(self, input, **kwargs):
        """Generate text using hfLLM transformers with XGrammar."""
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)
        
        # Get parameters
        format_schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, format_schema, tools, model, **kwargs)
        
        # Convert tools if provided (tools not directly supported in basic transformers)
        if tools:
            tools = self._convert_function_to_tools(tools)
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            input, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        model_inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
        
        # Set generation parameters
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': self._tokenizer.eos_token_id,
        }
        
        result = {"think": "", "content": "", "tool_calls": []}
        
        # Handle structured output with XGrammar
        if format_schema:
            try:
                # Setup XGrammar for structured generation
                tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                    self._tokenizer, 
                    vocab_size=self._config.vocab_size
                )
                grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
                
                # Compile grammar based on format type
                if hasattr(format_schema, 'model_json_schema'):
                    # Pydantic model
                    compiled_grammar = grammar_compiler.compile_json_schema(format_schema)
                elif isinstance(format_schema, str):
                    # JSON schema string
                    compiled_grammar = grammar_compiler.compile_json_schema(format_schema)
                else:
                    # Assume built-in JSON grammar
                    compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
                
                # Initialize grammar matcher
                matcher = xgr.GrammarMatcher(compiled_grammar)
                token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
                
                # Generate with grammar constraints
                generated_tokens = []
                input_ids = model_inputs.input_ids[0]
                
                for _ in range(generation_kwargs['max_new_tokens']):
                    # Get logits from model
                    with torch.no_grad():
                        outputs = self._model(input_ids.unsqueeze(0))
                        logits = outputs.logits[0, -1, :].float()  # Get last token logits
                    
                    # Apply grammar constraints
                    matcher.fill_next_token_bitmask(token_bitmask)
                    xgr.apply_token_bitmask_inplace(logits, token_bitmask.to(logits.device))
                    
                    # Sample next token
                    if generation_kwargs.get('do_sample', True):
                        probs = torch.softmax(logits / generation_kwargs.get('temperature', 1.0), dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    else:
                        next_token = torch.argmax(logits).item()
                    
                    # Accept token in grammar matcher
                    if not matcher.accept_token(next_token):
                        break
                    
                    generated_tokens.append(next_token)
                    input_ids = torch.cat([input_ids, torch.tensor([next_token], device=self._device)])
                    
                    # Check if we should stop
                    if next_token == self._tokenizer.eos_token_id or matcher.is_terminated():
                        break
                
                # Decode generated tokens
                generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
            except Exception as e:
                print(f"XGrammar error: {e}. Falling back to regular generation.")
                # Fallback to regular generation
                generated_ids = self._model.generate(
                    **model_inputs,
                    **generation_kwargs
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                generated_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            # Regular generation without grammar constraints
            generated_ids = self._model.generate(
                **model_inputs,
                **generation_kwargs
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            generated_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Extract thinking content from <think> tags
        think, content = self._extract_thinking(generated_text)
        result['think'] = think
        result['content'] = content
        
        # Tools are not directly supported but can be parsed from content
        # For now, return empty tool_calls
        result['tool_calls'] = []
        
        return result

    def _stream_chat(self, messages, format_schema, tools, model, **kwargs):
        """Generate streaming text using hfLLM transformers."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        model_inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
        
        # Set generation parameters
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': self._tokenizer.eos_token_id,
        }
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Start generation in separate thread
        generation_kwargs.update({
            'input_ids': model_inputs['input_ids'],
            'streamer': streamer,
        })
        
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens as they're generated
        def stream_generator():
            for new_text in streamer:
                yield {
                    "think": "",
                    "content": new_text,
                    "tool_calls": []
                }
            thread.join()
        
        return stream_generator()


# Test classes
class MenuItem(BaseModel):
    """A menu item in a restaurant."""
    course_name: str
    is_vegetarian: bool


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""
    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


class Person(BaseModel):
    """A person with basic information."""
    name: str
    age: int
    is_available: bool


class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    friends: List[FriendInfo]


# Test functions for tools
def get_current_time(timezone: str) -> dict:
    """Get the current time for a specific timezone"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
    }


def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    # Mock weather data - in real use case, call actual weather API
    import random
    temperatures = [15, 18, 22, 25, 28, 30, 33]
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    
    return {
        "city": city,
        "temperature": random.choice(temperatures),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80)
    }


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)


if __name__ == "__main__":
    # Initialize hfLLM LLM
    # NOTE: Make sure you have enough GPU memory for the model
    llm = hfLLM(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",  # Small model for testing
        #model="Qwen/Qwen3-0.6B",  # Small model for testing
        device=None,  # Let it auto-detect (cuda if available, else cpu)
    )
    
    # Test function calling with weather tool
    print("=== Testing basic chat ===")
    try:
        response = llm("Tell me a short joke about programming")
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error in basic chat: {e}")
    
    # Test structured output with XGrammar
    print("\n=== Testing XGrammar structured output ===")
    try:
        response = llm("Generate information about a person named Alice who is 25 years old", format=Person)
        print(f"Structured response: {response['content']}")
    except Exception as e:
        print(f"Error in structured output: {e}")
        print("Make sure XGrammar is installed: pip install xgrammar")
    
    # Test restaurant generation
    print("\n=== Testing restaurant generation ===")
    try:
        response = llm("Generate a restaurant in Miami with 2 menu items", format=Restaurant)
        print(f"Restaurant response: {response['content']}")
    except Exception as e:
        print(f"Error in restaurant generation: {e}")
    
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
    
    # Test built-in JSON grammar
    print("\n=== Testing built-in JSON grammar ===")
    try:
        response = llm("Generate any valid JSON object with name and age fields", format="json")
        print(f"JSON response: {response['content']}")
    except Exception as e:
        print(f"Error in built-in JSON: {e}")
    
    print("\n=== Simple Usage Examples ===")
    print("llm('Hello')  # Basic chat")
    print("llm('Generate person', format=PersonSchema)  # Structured output with XGrammar") 
    print("llm('What weather?', tools=[weather_func])  # Function calling (parsed from content)")
    print("llm(messages)  # Multi-turn chat")
    print("llm(text, stream=True)  # Streaming")
    
    print("\nTo use hfLLM with XGrammar:")
    print("1. Install: pip install transformers torch xgrammar")
    print("2. Choose a model: microsoft/DialoGPT-medium (small) or larger models")
    print("3. Make sure you have enough GPU memory")
    print("4. Run this script!")
    
    print("\nXGrammar Features:")
    print("• Ensures valid JSON output")
    print("• Supports Pydantic models for structured generation")
    print("• Built-in JSON schema support")
    print("• Runtime safeguards for constrained decoding")