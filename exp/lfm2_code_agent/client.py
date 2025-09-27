"""OpenAI-like client for LFM2 model with tool support."""

import json
import re
from typing import List, Dict, Any, Optional, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from code_tools import read_file, write_file, delete_lines


class LFMClient:
    """OpenAI-like client for LFM2 model."""

    def __init__(self, model_id: str = "LiquidAI/LFM2-350M", device_map: str = "auto"):
        """Initialize the LFM client.

        Args:
            model_id: The model identifier to load.
            device_map: Device mapping for the model.
        """
        print(f"🚀 Loading {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Default tools
        self.tools = {
            "read_file": read_file,
            "write_file": write_file,
            "delete_lines": delete_lines,
        }

        print("✅ Model loaded successfully!")

    def add_tool(self, name: str, func: Callable):
        """Add a custom tool function.

        Args:
            name: Name of the tool.
            func: The function to add as a tool.
        """
        self.tools[name] = func

    def get_weather(self, location: str) -> str:
        """Mock weather function for testing.

        Args:
            location: The location to get weather for.

        Returns:
            Mock weather information.
        """
        return f"Weather in {location}: Sunny, 72°F"

    def execute_tool_call(self, tool_call_text: str) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_call_text: The tool call text to execute.

        Returns:
            The result of the tool call.
        """
        match = re.search(r'(\w+)\((.*?)\)', tool_call_text)
        if not match:
            return "Error: Invalid tool call format"

        func_name, args_str = match.groups()

        if func_name not in self.tools:
            return f"Unknown function: {func_name}"

        try:
            func = self.tools[func_name]

            if '=' in args_str:
                # Parse keyword arguments
                kwargs = {}
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        kwargs[key] = value
                return func(**kwargs)
            else:
                # Parse positional arguments
                args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
                return func(*args)

        except Exception as e:
            return f"Error executing {func_name}: {e}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_turns: int = 10,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 512,
        tools: Optional[List[Callable]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Create a chat completion with tool support.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_turns: Maximum number of turns for tool calling.
            temperature: Sampling temperature.
            min_p: Minimum probability for sampling.
            repetition_penalty: Repetition penalty.
            max_new_tokens: Maximum new tokens to generate.
            tools: Optional list of tool functions to use.
            verbose: Whether to print verbose output.

        Returns:
            Dict containing the response and metadata.
        """
        # Add weather tool for testing
        if "get_weather" not in self.tools:
            self.tools["get_weather"] = self.get_weather

        # Use provided tools or default tools (only read_file and write_file work)
        tool_functions = tools if tools else [read_file, write_file]

        conversation_history = []
        current_content = messages[-1]["content"] if messages else ""

        if verbose:
            print(f"💬 User: {current_content}")

        for turn in range(max_turns):
            if verbose:
                print(f"🤔 Thinking... (Turn {turn + 1})")

            # Prepare input - try with tools first, fallback without tools
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tool_functions,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                ).to(self.model.device)
            except ValueError as e:
                if verbose:
                    print(f"⚠️  Tools error: {e}, trying without tools")
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                ).to(self.model.device)

            # Generate response
            output = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )

            response = self.tokenizer.decode(
                output[0][input_ids.shape[1]:],
                skip_special_tokens=False
            )

            if verbose:
                print(f"📝 Raw response: {response[:200]}...")

            # Check for tool calls
            tool_call_pattern = r'<\|tool_call_start\|\>\[(.*?)\]<\|tool_call_end\|\>'
            tool_calls = re.findall(tool_call_pattern, response)

            if tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    if verbose:
                        print(f"🔧 Tool call: {tool_call}")

                    result = self.execute_tool_call(tool_call)
                    tool_results.append({
                        "tool_call": tool_call,
                        "result": result
                    })

                    if verbose:
                        if result is not None:
                            result_str = str(result)
                            result_preview = result_str[:100] + "..." if len(result_str) > 100 else result_str
                        else:
                            result_preview = "None"
                        print(f"📄 Result: {result_preview}")

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "content": f"Tool result for {tool_call}: {result}"
                    })

                conversation_history.extend(tool_results)
            else:
                # No more tool calls, extract final response
                clean_response = response.split('<|tool_call_start|>')[0].strip()
                clean_response = clean_response.replace('<|im_end|>', '').strip()

                if verbose and clean_response:
                    print(f"✨ Final response: {clean_response}")

                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": clean_response
                        }
                    }],
                    "tool_calls": conversation_history,
                    "turns": turn + 1
                }

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Max turns reached without completion"
                }
            }],
            "tool_calls": conversation_history,
            "turns": max_turns
        }


if __name__ == "__main__":
    # Test the client
    client = LFMClient()

    print("\n" + "="*50)
    print("Testing LFM Client")
    print("="*50)

    # Test 1: Simple chat
    print("\n🧪 Test 1: Simple greeting")
    response = client.chat_completion([
        {"role": "user", "content": "Hello! What's your name?"}
    ])

    # Test 2: Weather tool
    print("\n🧪 Test 2: Weather tool")
    response = client.chat_completion([
        {"role": "user", "content": "What's the weather like in New York?"}
    ])

    # Test 3: File operations
    print("\n🧪 Test 3: File operations")
    response = client.chat_completion([
        {"role": "user", "content": "Create a file called test.txt with content 'Hello World' and then read it back"}
    ])

    print("\n✅ All tests completed!")