from __future__ import annotations

import os
from typing import Generator, Dict,Union

import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import re
import json
import inspect
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, Optional, Callable, List, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel


def convert_func_to_oai_tool(func: Any) -> dict:
    """Convert function to OpenAI function-calling tool schema."""
    if not callable(func):
        raise TypeError("Expected a callable object")

    signature = inspect.signature(func)
    hints = get_type_hints(func)

    # Map Python types to JSON schema types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    properties = {}
    required = []

    for name, param in signature.parameters.items():
        if name == "self":
            continue

        param_type = hints.get(name, str)
        json_type = type_map.get(param_type, "string")

        properties[name] = {
            "type": json_type,
            "description": f"Parameter {name}"
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model="", api_key=None, tools=None, format=None, timeout=None, **kwargs):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout
        self._api_key = api_key
        # Store any provider-specific parameters
        self._extra_params = kwargs
        self.llm = self._load_llm()

    @abstractmethod
    def _load_llm(self):
        """Load the LLM client/model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def chat(self, input, **kwargs):
        """Generate text using the LLM. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using the LLM. Must be implemented by subclasses."""
        pass

    def __call__(self, input, **kwargs) -> dict:
        """Generate text using the LLM. Auto-batches if input is a list of strings/prompts."""
        # Auto-batch only for list of strings or list of lists
        if isinstance(input, list) and input:
            # List of strings - batch processing
            if isinstance(input[0], str):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of lists - batch processing
            elif isinstance(input[0], list):
                max_workers = kwargs.pop('max_workers', len(input))
                return self.batch_call(input, max_workers=max_workers, **kwargs)
            # List of dicts - treat as single conversation, not batch
            elif isinstance(input[0], dict):
                return self.chat(input=input, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")

        return self.chat(input=input, **kwargs)

    def batch_call(self, inputs, max_workers=4, **kwargs):
        """Process multiple inputs in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda text: self(text, **kwargs), inputs))
        return results

    def _normalize_input(self, input):
        """Convert string input to message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            if all(isinstance(msg, dict) for msg in input):
                # Already in message format
                return input
            else:
                # List of strings, convert to user messages
                return [{"role": "user", "content": str(item)} for item in input]
        else:
            return [{"role": "user", "content": str(input)}]

    def _check_model(self, kwargs, default_model):
        """Check if model is provided, raise error if not."""
        model = kwargs.get("model") or default_model
        if model is None:
            raise ValueError("model is None")
        return model

    def _get_tools(self, kwargs):
        """Get tools from kwargs or use default tools."""
        return kwargs.pop('tools', None) or self._tools

    def _get_format(self, kwargs):
        """Get format from kwargs or use default format and convert to appropriate schema."""
        format_schema = kwargs.get('format', None) or self._format
        if not format_schema:
            return None

        return format_schema

    def _get_timeout(self, kwargs):
        """Get timeout from kwargs or use default timeout."""
        timeout = kwargs.get('timeout', None) or self._timeout
        if 'timeout' in kwargs:
            kwargs.pop("timeout")
        return timeout

    def _convert_function_to_tools(self, func: Optional[List[Callable]]) -> List[dict]:
        """Convert functions to tool format."""
        if not func:
            return []
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]

    def _extract_thinking(self, content):
        """Extract thinking content from <think> tags."""
        import re
        think = ''
        if '<think>' in content:
            if '</think>' in content:
                # Complete thinking block
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    think = think_match.group(1).strip()
                    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
            else:
                # Incomplete thinking block - extract everything after <think>
                think_match = re.search(r'<think>(.*)', content, re.DOTALL)
                if think_match:
                    think = think_match.group(1).strip()
                    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        return think, content


class LLM(BaseLLM):
    """HTTP client for FlowGen API server that inherits from BaseLLM."""

    def __init__(self, base_url: str | None = None, **kwargs):
        # priority: arg > env > default
        if base_url is None:
            base_url = os.getenv("BASE_URL", "http://0.0.0.0:8000")
        self._base_url = base_url.rstrip('/')
        super().__init__(**kwargs)

    def _load_llm(self):
        """No actual LLM to load - we're making HTTP requests."""
        return None

    def chat(self, input, **kwargs):
        """Generate text using HTTP API call to /chat endpoint."""
        input = self._normalize_input(input)

        # Get parameters
        format_schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)

        # Convert tools if provided
        if tools:
            tools = self._convert_function_to_tools(tools)

        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, format_schema, tools, **kwargs)

        # Prepare request payload
        payload = {
            "messages": input,
            "tools": tools,
            "options": {
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.8),
                "top_p": kwargs.get("top_p", 0.95),
                "stream": False
            }
        }

        # Add response format if provided
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                # Pydantic model
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema.model_json_schema()
                }
            elif isinstance(format_schema, dict):
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema
                }

        try:
            response = requests.post(
                f"{self._base_url}/chat",
                json=payload,
                timeout=timeout or 30
            )
            response.raise_for_status()

            # Convert API response to flowgen format
            api_result = response.json()
            return self._convert_api_response(api_result)

        except requests.exceptions.RequestException as e:
            return {"think": "", "content": f"API Error: {str(e)}", "tool_calls": []}
        except Exception as e:
            return {"think": "", "content": f"Error: {str(e)}", "tool_calls": []}

    def _stream_chat(self, messages, format_schema, tools, **kwargs):
        """Generate streaming text using HTTP API."""
        payload = {
            "messages": messages,
            "tools": tools,
            "options": {
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.8),
                "top_p": kwargs.get("top_p", 0.95),
                "stream": True
            }
        }

        # Add response format if provided
        if format_schema:
            if hasattr(format_schema, 'model_json_schema'):
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema.model_json_schema()
                }
            elif isinstance(format_schema, dict):
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema
                }

        def stream_generator():
            try:
                response = requests.post(
                    f"{self._base_url}/chat",
                    json=payload,
                    stream=True,
                    timeout=self._timeout or 30
                )
                response.raise_for_status()

                content_parts = []
                tool_calls = []

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data = line_str[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data)
                                if 'content' in chunk_data:
                                    content = chunk_data['content']
                                    if content is not None:
                                        content_parts.append(content)
                                        # Yield chunk like ollama/vllm do
                                        yield {"think": "", "content": content, "tool_calls": []}
                                elif 'completion' in chunk_data:
                                    # Handle final completion with tool calls
                                    completion = chunk_data['completion']
                                    final_result = self._convert_api_response(completion)
                                    if final_result['tool_calls']:
                                        yield {"think": "", "content": "", "tool_calls": final_result['tool_calls']}
                            except json.JSONDecodeError:
                                continue

            except requests.exceptions.RequestException as e:
                yield {"think": "", "content": f"Stream Error: {str(e)}", "tool_calls": []}

        return stream_generator()

    def _convert_api_response(self, api_result):
        """Convert API ChatCompletion format to flowgen LLM format."""
        result = {"think": "", "content": "", "tool_calls": []}

        # Handle error responses
        if "error" in api_result:
            result["content"] = f"API Error: {api_result['error']}"
            return result

        # Extract content from choices
        if "choices" in api_result and api_result["choices"]:
            message = api_result["choices"][0].get("message", {})
            content = message.get("content", "")

            # Extract thinking from content if present
            think, content = self._extract_thinking(content)
            result["think"] = think
            result["content"] = content

        # Extract tool calls
        if "tool_calls" in api_result and api_result["tool_calls"]:
            for tool_call in api_result["tool_calls"]:
                result["tool_calls"].append({
                    "id": f"call_{len(result['tool_calls'])}",
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"]) if isinstance(tool_call["arguments"], dict) else
                        tool_call["arguments"]
                    }
                })

        return result

class Agent:
    """Agentic wrapper that adds autonomous tool-calling behavior to any LLM.

    Provides transparent, streaming tool execution loops that work with any BaseLLM.
    Executes immediately when called, returning results directly.
    """

    def __init__(self, system_prompt: Optional[str] = None, llm: Optional[BaseLLM] = None,
                 tools: Optional[List[Callable]] = None, max_iterations: int = 25,
                 stream: bool = False, history: Optional[List[Dict]] = None,
                 enable_rich_debug: bool = True):
        """Initialize Agent with optional system prompt and LLM.

        Args:
            system_prompt: System prompt for the agent. If provided, automatically adds to history.
            llm: Any BaseLLM instance. If None, creates LLM() automatically.
            tools: List of callable functions to use as tools
            max_iterations: Maximum number of tool-calling iterations
            stream: If True, yields intermediate results during execution
            history: Previous conversation messages to continue from
            enable_rich_debug: If True, prints rich debug panels during execution
        """
        # Auto-create LLM if none provided
        self.llm = llm if llm is not None else LLM()
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.stream = stream
        self.history = history or []  # Store conversation history
        self.enable_rich_debug = enable_rich_debug
        self._tool_functions = {tool.__name__: tool for tool in self.tools}
        self._conversation = []  # Current conversation (gets reset on new calls)
        self._console = Console()

        # Add system prompt to history if provided
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def _run_sync(self, input: Union[str, List[Dict]], **kwargs) -> Union[Dict, Generator]:
        """Execute agentic behavior synchronously for sync LLMs."""
        if self.stream or kwargs.get('stream', False):
            return self._stream_execute_sync(input, **kwargs)
        else:
            return self._execute_sync(input, **kwargs)

    def __call__(self, input: Union[str, List[Dict]], **kwargs):
        """Execute agent call and return result directly."""
        return self._run_sync(input, **kwargs)

    def _execute_sync(self, input: Union[str, List[Dict]], **kwargs) -> Dict:
        """Non-streaming execution for sync LLMs - returns final result."""
        messages = self._normalize_input(input)
        # Prepend history to current conversation
        messages = self.history + messages
        self._conversation = messages.copy()

        # Check for rich debug override
        enable_debug = kwargs.pop('enable_rich_debug', self.enable_rich_debug)

        # Debug: Show user message
        if enable_debug and messages:
            user_msg = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), None)
            if user_msg:
                self._console.print(Panel(
                    user_msg.get('content', ''),
                    title="User",
                    title_align='left',
                    border_style="blue",
                    padding=(0, 1),
                    expand=False
                ))

        # Override LLM tools with agent tools if provided
        if self.tools:
            kwargs['tools'] = self.tools

        for iteration in range(self.max_iterations):
            response = self.llm(messages, **kwargs)

            # Debug: Show assistant response
            if enable_debug:
                content = response.get('content', '')
                think = response.get('think', '')
                debug_text = ""
                if think:
                    debug_text += f"Think: {think}\\n\\n"
                if content:
                    debug_text += f"Response: {content}"

                # Use Markdown for better formatting of assistant response
                markdown_content = Markdown(content) if content else ""
                self._console.print(Panel(
                    markdown_content,
                    title="Assistant",
                    title_align='center',
                    border_style="green",
                    padding=(0, 1),
                    expand=False
                ))

            # No tool calls - add assistant response and return final result
            if not response.get('tool_calls'):
                # Add the assistant's response to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.get('content', '')
                })

                # Update conversation history
                self._conversation = messages

                return {
                    'content': response.get('content', ''),
                    'think': response.get('think', ''),
                    'iterations': iteration + 1,
                    'messages': messages,
                    'final': True
                }

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "tool_calls": response['tool_calls']
            })

            # Execute tools and add results
            for i, tool_call in enumerate(response['tool_calls']):
                # Store original tool call for ID
                original_tool_call = tool_call
                # Debug: Show tool call
                tool_call_func = tool_call['function']
                if enable_debug:
                    tool_info = f"Tool: {tool_call_func['name']}("
                    if tool_call_func.get('arguments'):
                        tool_info += f"{json.dumps(tool_call_func['arguments'])}"
                    else:
                        tool_info += ")"

                    self._console.print(Panel(
                        tool_info.strip(),
                        title="Tool Call",
                        title_align='right',
                        border_style="yellow",
                        padding=(0, 1),
                        expand=False
                    ))

                tool_result = self._execute_tool_sync(tool_call_func)

                # Debug: Show tool result
                if enable_debug:
                    result_text = str(tool_result)
                    # Handle long text by removing width restriction and improving text handling
                    self._console.print(Panel(
                        result_text,
                        title=f"Tool Result ({tool_call_func['name']})",
                        title_align='right',
                        border_style="cyan",
                        padding=(0, 1),
                        expand=False
                    ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": original_tool_call.get('id', f"call_{i}"),
                    "name": tool_call_func['name'],
                    "content": str(tool_result)
                })

        # Update conversation history
        self._conversation = messages

        # Max iterations reached
        return {
            'content': 'Max iterations reached',
            'think': '',
            'iterations': self.max_iterations,
            'messages': messages,
            'final': True,
            'truncated': True
        }

    def _stream_execute_sync(self, input: Union[str, List[Dict]], **kwargs) -> Generator[Dict, None, None]:
        """Streaming execution for sync LLMs - yields intermediate results and token-level streaming."""
        messages = self._normalize_input(input)
        # Prepend history to current conversation
        messages = self.history + messages
        self._conversation = messages.copy()

        if self.tools:
            kwargs['tools'] = self.tools

        for iteration in range(self.max_iterations):
            # Yield iteration start - always include current messages
            yield {
                'type': 'iteration_start',
                'iteration': iteration + 1,
                'messages_count': len(messages),
                'messages': messages.copy()
            }

            # Try streaming first, fallback to regular if it fails
            response = None

            try:
                # Try streaming if requested
                kwargs_for_llm = kwargs.copy()
                kwargs_for_llm['stream'] = True

                llm_stream = self.llm(messages, **kwargs_for_llm)

                # Check if we got a streaming response from any LLM (vLLM, Gemini, hgLLM)
                if hasattr(llm_stream, '__class__') and 'Stream' in str(llm_stream.__class__):
                    # Streaming response from OpenAI-compatible APIs (vLLM, Gemini, etc.)
                    accumulated_content = ""
                    accumulated_think = ""
                    tool_calls_dict = {}

                    for chunk in llm_stream:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            choice = chunk.choices[0]

                            if hasattr(choice, 'delta') and choice.delta:
                                # Content streaming
                                if hasattr(choice.delta, 'content') and choice.delta.content:
                                    token = choice.delta.content
                                    accumulated_content += token

                                    yield {
                                        'type': 'token',
                                        'content': token,
                                        'accumulated_content': accumulated_content,
                                        'iteration': iteration + 1,
                                        'messages': messages.copy()
                                    }

                                # Tool calls streaming - handle both vLLM and Gemini patterns
                                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                                    for tool_call in choice.delta.tool_calls:
                                        if hasattr(tool_call, 'index') and tool_call.index is not None:
                                            # vLLM pattern: incremental streaming with index
                                            idx = tool_call.index
                                            if idx not in tool_calls_dict:
                                                tool_calls_dict[idx] = {
                                                    'id': getattr(tool_call, 'id', f'call_{idx}'),
                                                    'type': 'function',
                                                    'function': {'name': '', 'arguments': ''}
                                                }

                                            if hasattr(tool_call, 'function') and tool_call.function:
                                                if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                                    tool_calls_dict[idx]['function']['name'] = tool_call.function.name
                                                if hasattr(tool_call.function,
                                                           'arguments') and tool_call.function.arguments:
                                                    tool_calls_dict[idx]['function'][
                                                        'arguments'] += tool_call.function.arguments
                                        else:
                                            # Gemini pattern: complete tool calls in chunks
                                            if hasattr(tool_call, 'function') and tool_call.function:
                                                idx = len(tool_calls_dict)
                                                tool_calls_dict[idx] = {
                                                    'id': getattr(tool_call, 'id', f'call_{idx}'),
                                                    'type': getattr(tool_call, 'type', 'function'),
                                                    'function': {
                                                        'name': tool_call.function.name or '',
                                                        'arguments': tool_call.function.arguments or '{}'
                                                    }
                                                }

                    # Build final response from streaming
                    response = {
                        'content': accumulated_content,
                        'think': accumulated_think,
                        'tool_calls': list(tool_calls_dict.values()) if tool_calls_dict else []
                    }

                    # Ensure tool call arguments are valid JSON
                    for tool_call in response['tool_calls']:
                        if not tool_call['function']['arguments']:
                            tool_call['function']['arguments'] = '{}'

                elif hasattr(llm_stream, '__iter__') and not isinstance(llm_stream, (str, dict)):
                    # Custom streaming format (LLM/hgLLM style)
                    accumulated_content = ""
                    accumulated_think = ""
                    current_buffer = ""
                    in_thinking = False
                    thinking_content = ""
                    in_tool_call = False
                    tool_call_content = ""
                    tool_calls_found = []

                    for chunk in llm_stream:
                        if isinstance(chunk, str):
                            # Raw token streaming
                            current_buffer += chunk
                        elif isinstance(chunk, dict):
                            # LLM chunk format
                            if 'content' in chunk:
                                token = chunk.get('content', '')
                                current_buffer += token

                        # Process buffer for thinking and tool call tags
                        while True:
                            processed = False

                            if not in_thinking and not in_tool_call:
                                # Check for thinking tag
                                think_match = re.search(r'<think>', current_buffer)
                                if think_match:
                                    # Yield content before thinking
                                    before_think = current_buffer[:think_match.start()]
                                    if before_think:
                                        accumulated_content += before_think
                                        yield {
                                            'type': 'token',
                                            'content': before_think,
                                            'accumulated_content': accumulated_content,
                                            'iteration': iteration + 1
                                        }

                                    current_buffer = current_buffer[think_match.end():]
                                    in_thinking = True
                                    thinking_content = ""
                                    processed = True
                                    continue

                                # Check for tool_call tag
                                tool_match = re.search(r'<tool_call>', current_buffer)
                                if tool_match:
                                    # Yield content before tool call
                                    before_tool = current_buffer[:tool_match.start()]
                                    if before_tool:
                                        accumulated_content += before_tool
                                        yield {
                                            'type': 'token',
                                            'content': before_tool,
                                            'accumulated_content': accumulated_content,
                                            'iteration': iteration + 1
                                        }

                                    current_buffer = current_buffer[tool_match.end():]
                                    in_tool_call = True
                                    tool_call_content = ""
                                    processed = True
                                    continue

                                # No special tags - yield as regular content
                                if current_buffer and not re.search(r'<t?h?i?n?k?>?$|<t?o?o?l?_?c?a?l?l?>?$',
                                                                    current_buffer):
                                    accumulated_content += current_buffer
                                    yield {
                                        'type': 'token',
                                        'content': current_buffer,
                                        'accumulated_content': accumulated_content,
                                        'iteration': iteration + 1
                                    }
                                    current_buffer = ""

                            elif in_thinking:
                                # Look for thinking end tag
                                end_match = re.search(r'</think>', current_buffer)
                                if end_match:
                                    thinking_content += current_buffer[:end_match.start()]
                                    current_buffer = current_buffer[end_match.end():]
                                    in_thinking = False

                                    if thinking_content.strip():
                                        accumulated_think += thinking_content
                                        yield {
                                            'type': 'think_block',
                                            'content': thinking_content.strip(),
                                            'accumulated_think': accumulated_think,
                                            'iteration': iteration + 1
                                        }

                                    thinking_content = ""
                                    processed = True
                                    continue
                                else:
                                    # Accumulate thinking content and yield as think tokens
                                    if current_buffer and not re.search(r'</t?h?i?n?k?>?$', current_buffer):
                                        thinking_content += current_buffer
                                        yield {
                                            'type': 'think_token',
                                            'content': current_buffer,
                                            'accumulated_think': thinking_content,
                                            'iteration': iteration + 1
                                        }
                                        current_buffer = ""

                            elif in_tool_call:
                                # Look for tool call end tag
                                end_match = re.search(r'</tool_call>', current_buffer)
                                if end_match:
                                    tool_call_content += current_buffer[:end_match.start()]
                                    current_buffer = current_buffer[end_match.end():]
                                    in_tool_call = False

                                    # Parse tool call
                                    if tool_call_content.strip():
                                        try:
                                            tool_data = json.loads(tool_call_content.strip())
                                            tool_name = tool_data.get("name", "unknown")
                                            tool_args = tool_data.get("arguments", {})

                                            yield {
                                                'type': 'tool_start',
                                                'tool_name': tool_name,
                                                'tool_args': tool_args,
                                                'iteration': iteration + 1
                                            }

                                            # Execute tool
                                            tool_result = self._execute_tool_sync({
                                                'name': tool_name,
                                                'arguments': json.dumps(tool_args) if isinstance(tool_args,
                                                                                                 dict) else tool_args
                                            })

                                            yield {
                                                'type': 'tool_result',
                                                'tool_name': tool_name,
                                                'tool_args': tool_args,
                                                'result': str(tool_result),
                                                'iteration': iteration + 1
                                            }

                                            # Add to tool calls for response
                                            tool_calls_found.append({
                                                'id': f'call_{len(tool_calls_found)}',
                                                'type': 'function',
                                                'function': {
                                                    'name': tool_name,
                                                    'arguments': json.dumps(tool_args) if isinstance(tool_args,
                                                                                                     dict) else tool_args
                                                }
                                            })

                                        except Exception as e:
                                            yield {
                                                'type': 'tool_result',
                                                'tool_name': 'error',
                                                'tool_args': {},
                                                'result': f"Failed to parse/execute tool: {str(e)}",
                                                'iteration': iteration + 1
                                            }

                                    tool_call_content = ""
                                    processed = True
                                    continue
                                else:
                                    # Accumulate tool call content
                                    if current_buffer and not re.search(r'</t?o?o?l?_?c?a?l?l?>?$', current_buffer):
                                        tool_call_content += current_buffer
                                        current_buffer = ""

                            if not processed:
                                break

                    response = {
                        'content': accumulated_content,
                        'think': accumulated_think,
                        'tool_calls': tool_calls_found,
                        '_tools_executed_in_streaming': True  # Flag to prevent double execution
                    }

                else:
                    # Not a streaming response, use as regular response
                    response = llm_stream

            except Exception as e:
                # Fallback to non-streaming if streaming fails
                try:
                    kwargs_fallback = kwargs.copy()
                    kwargs_fallback.pop('stream', None)
                    response = self.llm(messages, **kwargs_fallback)
                except Exception as fallback_e:
                    # If both fail, raise the original streaming error
                    raise e

            # Yield LLM response - always include current messages
            yield {
                'type': 'llm_response',
                'content': response.get('content', ''),
                'think': response.get('think', ''),
                'tool_calls': response.get('tool_calls', []),
                'iteration': iteration + 1,
                'messages': messages.copy()
            }

            # No tool calls - add assistant response and yield final result
            if not response.get('tool_calls'):
                # Add the assistant's response to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.get('content', '')
                })

                # Update conversation history
                self._conversation = messages

                yield {
                    'type': 'final',
                    'content': response.get('content', ''),
                    'think': response.get('think', ''),
                    'iterations': iteration + 1,
                    'messages': messages.copy()
                }
                return

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "tool_calls": response['tool_calls']
            })

            # Execute tools and yield results (only if not already executed during streaming)
            if not response.get('_tools_executed_in_streaming', False):
                for i, tool_call in enumerate(response['tool_calls']):
                    tool_func = tool_call['function']
                    yield {
                        'type': 'tool_start',
                        'tool_name': tool_func['name'],
                        'tool_args': tool_func['arguments'],
                        'iteration': iteration + 1,
                        'messages': messages.copy()
                    }

                    tool_result = self._execute_tool_sync(tool_func)

                    yield {
                        'type': 'tool_result',
                        'tool_name': tool_func['name'],
                        'tool_args': tool_func['arguments'],
                        'result': str(tool_result),
                        'iteration': iteration + 1,
                        'messages': messages.copy()
                    }

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get('id', f"call_{i}"),
                        "name": tool_func['name'],
                        "content": str(tool_result)
                    })
            else:
                # Tools were already executed during streaming, just add results to messages
                for i, tool_call in enumerate(response['tool_calls']):
                    tool_func = tool_call['function']
                    # Tool result should be available from streaming execution
                    # For now, just add a placeholder - in real implementation,
                    # we'd track the results from streaming
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get('id', f"call_{i}"),
                        "name": tool_func['name'],
                        "content": "Tool executed during streaming"
                    })

        # Update conversation history
        self._conversation = messages

        # Max iterations reached
        yield {
            'type': 'final',
            'content': 'Max iterations reached',
            'iterations': self.max_iterations,
            'messages': messages.copy(),
            'truncated': True
        }

    def _normalize_input(self, input: Union[str, List[Dict]]) -> List[Dict]:
        """Convert input to message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        return input.copy()  # Don't modify original

    def _format_tool_calls(self, tools: List[Dict]) -> List[Dict]:
        """Format tool calls for OpenAI message format."""
        return [
            {
                "id": tool.get('id', f"call_{i}"),
                "function": {
                    "name": tool['name'],
                    "arguments": json.dumps(tool['arguments']) if isinstance(tool['arguments'], dict) else tool[
                        'arguments']
                },
                "type": "function"
            }
            for i, tool in enumerate(tools)
        ]

    def _execute_tool_sync(self, tool_call: Dict) -> Any:
        """Execute a single tool call synchronously."""
        tool_name = tool_call['name']
        tool_args = json.loads(str(tool_call['arguments']))

        if tool_name not in self._tool_functions:
            return f"Error: Tool '{tool_name}' not found"

        try:
            return self._tool_functions[tool_name](**tool_args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def add_tool(self, tool: Callable) -> None:
        """Add a tool function to the agent."""
        self.tools.append(tool)
        self._tool_functions[tool.__name__] = tool

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool by name."""
        self.tools = [t for t in self.tools if t.__name__ != tool_name]
        self._tool_functions.pop(tool_name, None)

    def get_conversation(self) -> List[Dict]:
        """Get the current conversation history."""
        return self._conversation.copy()

    def add_history(self, messages: List[Dict]) -> None:
        """Add messages to the conversation history."""
        self.history.extend(messages)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        self._conversation = []

    def set_history(self, messages: List[Dict]) -> None:
        """Replace current history with new messages."""
        self.history = messages.copy()

    def __rshift__(self, other: 'Agent') -> 'AgentChain':
        """Chain agents using >> operator."""
        return AgentChain([self, other])

    def export(self, format: str = 'dict') -> Union[Dict, str]:
        """Export agent state and conversation in various formats.

        Args:
            format: 'dict', 'json', or 'markdown'
        """
        data = {
            'conversation': self._conversation,
            'history': self.history,
            'tools': [tool.__name__ for tool in self.tools],
            'max_iterations': self.max_iterations,
            'model_info': getattr(self.llm, '_model', 'unknown'),
            'stream': self.stream
        }

        if format == 'dict':
            return data
        elif format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'markdown':
            return self._export_markdown(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, data: Union[str, Dict], llm: BaseLLM, tools: Optional[List[Callable]] = None) -> 'Agent':
        """Load Agent from exported JSON/dict data.

        Args:
            data: JSON string or dict from agent.export()
            llm: LLM instance to use (must be provided)
            tools: List of tool functions (must match exported tool names)

        Returns:
            Agent instance restored from the data
        """
        # Parse JSON if string
        if isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            parsed_data = data.copy()

        # Validate tools match
        exported_tool_names = set(parsed_data.get('tools', []))
        provided_tool_names = set(tool.__name__ for tool in (tools or []))

        if exported_tool_names and not exported_tool_names.issubset(provided_tool_names):
            missing = exported_tool_names - provided_tool_names
            raise ValueError(f"Missing tools for loading: {missing}")

        # Create agent with restored state
        agent = cls(
            llm=llm,
            tools=tools,
            max_iterations=parsed_data.get('max_iterations', 10),
            stream=parsed_data.get('stream', False),
            history=parsed_data.get('history', [])
        )

        # Restore conversation state
        agent._conversation = parsed_data.get('conversation', [])

        return agent

    def _export_markdown(self, data: Dict) -> str:
        """Export conversation as markdown format."""
        md = f"# Agent Conversation Export\n\n"
        md += f"**Model:** {data['model_info']}\n"
        md += f"**Tools:** {', '.join(data['tools'])}\n"
        md += f"**Max Iterations:** {data['max_iterations']}\n\n"

        # Export conversation
        md += "## Conversation\n\n"
        for msg in data['conversation']:
            role = msg.get('role', 'unknown').title()
            content = msg.get('content', '')

            if role == 'System':
                md += f"**🤖 System:** {content}\n\n"
            elif role == 'User':
                md += f"**👤 User:** {content}\n\n"
            elif role == 'Assistant':
                if 'tool_calls' in msg:
                    md += f"**🤖 Assistant:** *Called tools: {', '.join([tc['function']['name'] for tc in msg['tool_calls']])}\n\n"
                else:
                    md += f"**🤖 Assistant:** {content}\n\n"
            elif role == 'Tool':
                tool_name = msg.get('name', 'unknown')
                md += f"**⚡ Tool ({tool_name}):** {content}\n\n"

        return md


class AgentChain:
    """Chain multiple agents together using >> operator."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def __rshift__(self, other: Agent) -> 'AgentChain':
        """Add another agent to the chain."""
        return AgentChain(self.agents + [other])

    def __call__(self, input: Union[str, List[Dict]], **kwargs) -> Dict:
        """Execute the agent chain sequentially, passing full conversation history."""
        current_input = input
        results = []
        accumulated_conversation = []

        for i, agent in enumerate(self.agents):
            if i == 0:
                # For first agent, use original input
                result = agent(current_input, **kwargs)
            else:
                # For subsequent agents, pass the accumulated conversation history
                # This gives them full context of the entire chain so far
                current_input = result.get('content', '')
                result = agent(current_input, history=accumulated_conversation, **kwargs)

            # Result is already executed, no need to convert

            results.append(result)

            # If streaming was requested, we can't chain properly
            if (hasattr(result, '__iter__') and
                not isinstance(result, (str, dict)) and
                not hasattr(result, 'get')):  # Also allow dict-like objects
                raise ValueError("Cannot chain streaming agents. Set stream=False.")

            # Accumulate conversation history for next agent
            # Get the full conversation from this agent's execution
            agent_conversation = agent.get_conversation()
            if agent_conversation:
                accumulated_conversation = agent_conversation

        # Return final result with chain metadata
        final_result = results[-1].copy()
        final_result['chain_results'] = results
        final_result['chain_length'] = len(self.agents)
        final_result['full_conversation'] = accumulated_conversation

        return final_result

    def export(self, format: str = 'dict') -> Union[Dict, str]:
        """Export entire chain conversation."""
        chain_data = {
            'chain_length': len(self.agents),
            'agents': []
        }

        for i, agent in enumerate(self.agents):
            agent_data = agent.export('dict')
            agent_data['position'] = i
            chain_data['agents'].append(agent_data)

        if format == 'dict':
            return chain_data
        elif format == 'json':
            return json.dumps(chain_data, indent=2)
        elif format == 'markdown':
            md = f"# Agent Chain Export ({len(self.agents)} agents)\n\n"
            for i, agent in enumerate(self.agents):
                md += f"## Agent {i + 1}\n\n"
                md += agent._export_markdown(agent.export('dict'))
                md += "\n---\n\n"
            return md
        else:
            raise ValueError(f"Unsupported format: {format}")

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



def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)  # Note: eval is unsafe, use a proper parser in production
        return str(result)
    except:
        return "Invalid expression"


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [Mock results]"


"""
curl --location 'http://0.0.0.0:8000/load' \
--header 'Content-Type: application/json' \
--data '{
  "model_path": "/home/ntlpt59/Downloads/LFM2-350M-F16.gguf",
  "tokenizer_name": "LiquidAI/LFM2-350M",
  "options": {
    "n_gpu_layers": -1,
    "n_ctx": 2048,
    "n_batch": 256
  }
}'
"""


def tool_calling_transformers():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")

    def get_current_temperature(location: str):
        """
        Gets the temperature at a given location.

        Args:
            location: The location to get the temperature for, in the format "city, country"
        """
        return 22.0  # bug: Sometimes the temperature is not 22. low priority to fix tho

    tools = [get_current_temperature]

    chat = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions or call tools."},
        {"role": "user", "content": "Hey, what's the weather like in Paris right now?"}
    ]

    # With add_generation_prompt=True
    tool_prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )
    print("==== tool_prompt (add_generation_prompt=True) ====")
    print(tool_prompt)

    # With add_generation_prompt=False
    tool_prompt_no_gen = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=False,
        tokenize=False
    )
    print("==== tool_prompt_no_gen (add_generation_prompt=False) ====")
    print(tool_prompt_no_gen)

    response="""<tool_call>
{"arguments": {"location": "Paris, France"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
"""

    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": {"location": "Paris, France"}
                }
            }
        ]
    }
    chat.append(message)


    tool_prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )
    print("==== tool_prompt (add_generation_prompt=True) ====")
    print(tool_prompt)

    # With add_generation_prompt=False
    tool_prompt_no_gen = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=False,
        tokenize=False
    )
    print("==== tool_prompt_no_gen (add_generation_prompt=False) ====")
    print(tool_prompt_no_gen)


def testing():
    llm = LLM()

    print("=== Non-Streaming Cases ===")
    # 1. Basic message
    result = llm("Hello, how are you?")
    print("[Non-Stream | Basic]", result)

    # 2. Structured output
    class PersonSchema(BaseModel):
        name: str
        age: int

    result = llm("My name is Alice and I am 28 years old",
                 format=PersonSchema)
    print("[Non-Stream | Structured]", result)

    # 3. Tool call
    result = llm("Add two numbers: a=5, b=7",tools=[add_two_numbers])
    print("[Non-Stream | Tool]", result)


    print("\n=== Streaming Cases ===")
    # 1. Basic message
    print("[Stream | Basic]")
    for event in llm("Stream hello world", stream=True):
        print(event, end="", flush=True)
    print("\n")

    # 2. Structured output
    print("[Stream | Structured]")
    for event in llm("My name is Bob and I am 35 years old",
                     format=PersonSchema,
                     stream=True):
        print(event, end="", flush=True)
    print("\n")

    # 3. Tool call
    print("[Stream | Tool]")
    stream_agent = Agent(llm=llm, tools=[add_two_numbers], stream=True)
    for event in stream_agent("Add two numbers: a=3, b=9", stream=True):
        if event['type'] == 'final':
            print("Final:", event['content'])
        elif event['type'] in ('token', 'tool_start', 'tool_result'):
            print(event)

if __name__ == '__main__':
    tool_calling_transformers()
