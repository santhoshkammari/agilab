from __future__ import annotations

import json
import re
from typing import Generator, Dict, Any, List, Callable, Optional, Union
from src.llm import BaseLLM
from src.llm import LLM
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown




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
                        tool_info+=")"
                    
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
                                                if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                                    tool_calls_dict[idx]['function']['arguments'] += tool_call.function.arguments
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
                
                elif hasattr(llm_stream, '__iter__'):
                    # Custom streaming format (generator). Handle dict-chunk streams (FlowGen)
                    # and raw text streams (tag-parsing) in a unified way.
                    accumulated_content = ""
                    accumulated_think = ""
                    current_buffer = ""
                    in_thinking = False
                    thinking_content = ""
                    in_tool_call = False
                    tool_call_content = ""
                    tool_calls_found = []
                    executed_during_stream = False
                    
                    for chunk in llm_stream:
                        if isinstance(chunk, str):
                            # Raw token streaming
                            current_buffer += chunk
                        elif isinstance(chunk, dict):
                            # LLM chunk format
                            if chunk.get('content'):
                                token = chunk.get('content', '')
                                current_buffer += token
                            # Some servers emit tool_calls as a final chunk
                            if chunk.get('tool_calls'):
                                # Normalize arguments to strings for consistency
                                for tc in chunk['tool_calls']:
                                    fn = tc.get('function', {})
                                    args = fn.get('arguments', '{}')
                                    if not isinstance(args, str):
                                        fn['arguments'] = json.dumps(args)
                                tool_calls_found.extend(chunk['tool_calls'])
                                # Do not execute here; allow the normal post-stream path to run tools
                                # Continue to process remaining chunks if any
                                continue
                        
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
                                if current_buffer and not re.search(r'<t?h?i?n?k?>?$|<t?o?o?l?_?c?a?l?l?>?$', current_buffer):
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
                                                'arguments': json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
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
                                                    'arguments': json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                                                }
                                            })
                                            executed_during_stream = True
                                            
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
                        '_tools_executed_in_streaming': executed_during_stream  # Flag to prevent double execution
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
                    "arguments": json.dumps(tool['arguments']) if isinstance(tool['arguments'],dict) else tool['arguments']
                },
                "type": "function"
            }
            for i, tool in enumerate(tools)
        ]
    
    def _execute_tool_sync(self, tool_call: Dict) -> Any:
        """Execute a single tool call synchronously."""
        tool_name = tool_call['name']
        raw_args = tool_call.get('arguments', '{}')
        if isinstance(raw_args, dict):
            tool_args = raw_args
        else:
            tool_args = json.loads(str(raw_args))

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
                md += f"## Agent {i+1}\n\n"
                md += agent._export_markdown(agent.export('dict'))
                md += "\n---\n\n"
            return md
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for quick agent creation
def create_agent(llm: BaseLLM, tools: Optional[List[Callable]] = None, **kwargs) -> Agent:
    """Create an Agent with any LLM and tools."""
    return Agent(llm, tools, **kwargs)


# Example usage and demo
if __name__ == '__main__':
    from ..llm.gemini import Gemini

    # Example tools
    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        return f"Weather in {location}: Sunny, 25°C"
    
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

    print("=== Agent Framework Demo ===\n")
    
    # Create LLM (can be any provider)
    llm = Gemini()  # or vLLM(...) or Ollama(...)
    
    # 1. Basic agent usage
    print("1. Basic agent with tools:")
    agent = Agent(llm=llm, tools=[get_weather, calculate])
    result = agent("What's 2+3 and weather in Paris?",enable_rich_debug=True)
    print(f"Result: {result['content'][:100]}...\n")



    # 2. History management
    print("2. History management:")
    # Continue from previous conversation
    history = [{"role": "system", "content": "You are a helpful assistant"}]
    agent_with_history = Agent(llm=llm, tools=[calculate], history=history,enable_rich_debug=True)
    
    # Add more history
    agent_with_history.add_history([
        {"role": "user", "content": "I'm working on math problems"},
        {"role": "assistant", "content": "I'll help you with calculations!"}
    ])
    
    result = agent_with_history("What's 10 * 15?",enable_rich_debug=True)
    print(f"Result with history: {result['content'][:50]}...\n")
    
    # 3. Agent chaining with >> operator
    print("3. Agent chaining:")
    research_agent = Agent(llm=llm, tools=[search_web])
    analysis_agent = Agent(llm=llm, tools=[calculate])
    summary_agent = Agent(llm=llm, tools=[])
    
    # Chain agents
    chain = research_agent >> analysis_agent >> summary_agent
    result = chain("Research Python popularity and analyze the numbers")
    print(f"Chain result: {result['content'][:100]}...")
    print(f"Chain length: {result['chain_length']}\n")
    
    # 4. Export in different formats
    print("4. Export conversation:")
    
    # JSON export
    json_export = agent.export('json')
    print(f"JSON export (first 100 chars): {json_export[:100]}...\n")
    
    # Markdown export for other agents
    md_export = agent.export('markdown')
    print("Markdown export:")
    print(md_export[:200] + "...\n")
    
    # Load agent from JSON
    print("Loading agent from JSON:")
    json_data = agent.export('json')
    restored_agent = Agent.load(json_data, llm=llm, tools=[get_weather, calculate])
    print(f"Restored agent has {len(restored_agent.history)} history items")
    print(f"Restored conversation: {len(restored_agent.get_conversation())} messages\n")
    
    # 5. Streaming with sync execution
    print("5. Streaming agent:")
    stream_agent = Agent(llm=llm, tools=[get_weather], stream=True)
    
    for event in stream_agent("Get weather for London", stream=True):
        if event['type'] == 'tool_start':
            print(f"🔧 {event['tool_name']}")
        elif event['type'] == 'final':
            print(f"🏁 {event['content'][:50]}...")
            break
    
    print("\n=== New Agent Features ===")
    print("✅ History Management:")
    print("   • agent = Agent(llm, tools, history=prev_messages)")
    print("   • agent.add_history([...]), agent.clear_history()")
    print("✅ Agent Chaining:")  
    print("   • research_agent >> analysis_agent >> summary_agent")
    print("   • result = chain('query') - automatic pipeline")
    print("✅ Export/Serialization:")
    print("   • agent.export('json') - for data storage")
    print("   • agent.export('markdown') - for other agents")
    print("   • Agent.load(json_data, llm, tools) - restore from export")
    print("   • Full conversation + metadata export")
    print("✅ Todo Management:")
    print("   • todo_write(todos) - structured task list management")
    print("   • Track task status: pending, in_progress, completed")
    print("   • Priority levels: high, medium, low")
    print("✅ All Previous Features:")
    print("   • Works with any sync BaseLLM, streaming, pluggable tools")
