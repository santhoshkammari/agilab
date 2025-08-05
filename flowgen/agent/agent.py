from __future__ import annotations

import asyncio
import json
from typing import Generator, Dict, Any, List, Callable, Optional, Union
from ..llm import BaseLLM
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown


class _AgentCall:
    """Awaitable wrapper that allows agent('hello') and await agent('hello') to both work."""
    
    def __init__(self, agent, input, **kwargs):
        self.agent = agent
        self.input = input
        self.kwargs = kwargs
        self._result = None
        self._executed = False
    
    def __await__(self):
        """Makes this object awaitable - used when: await agent('hello')"""
        async def _async_call():
            if self.agent._is_async_llm():
                return await self.agent.run_async(self.input, **self.kwargs)
            else:
                # For sync LLMs in async context, run in thread pool
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(
                        executor, 
                        lambda: self.agent._run_sync(self.input, **self.kwargs)
                    )
        
        return _async_call().__await__()
    
    def __getattr__(self, name):
        """Auto-execute on attribute access - used when: agent('hello').content"""
        if not self._executed:
            self._execute_sync()
        return getattr(self._result, name)
    
    def __getitem__(self, key):
        """Auto-execute on item access - used when: agent('hello')['content']"""
        if not self._executed:
            self._execute_sync()
        return self._result[key]
    
    def __iter__(self):
        """Auto-execute on iteration - used when: for x in agent('hello')"""
        if not self._executed:
            self._execute_sync()
        return iter(self._result)
    
    def __str__(self):
        """Auto-execute on string conversion - used when: str(agent('hello'))"""
        if not self._executed:
            self._execute_sync()
        return str(self._result)
    
    def __repr__(self):
        """Auto-execute on repr - used when: repr(agent('hello'))"""
        if not self._executed:
            self._execute_sync()
        return repr(self._result)
    
    def get(self, key, default=None):
        """Dict-like access - used when: agent('hello').get('content')"""
        if not self._executed:
            self._execute_sync()
        return self._result.get(key, default) if hasattr(self._result, 'get') else getattr(self._result, key, default)
    
    def _execute_sync(self):
        """Execute the agent call synchronously."""
        if self._executed:
            return self._result
        
        if self.agent._is_async_llm():
            raise RuntimeError("Cannot use sync access with async LLM. Use 'await agent(...)' instead.")
        
        self._result = self.agent._run_sync(self.input, **self.kwargs)
        self._executed = True
        return self._result


class Agent:
    """Agentic wrapper that adds autonomous tool-calling behavior to any LLM.
    
    Provides transparent, streaming tool execution loops that work with any BaseLLM.
    Can be used as a drop-in replacement for direct LLM calls.
    """
    
    def __init__(self, llm: BaseLLM, tools: Optional[List[Callable]] = None, 
                 max_iterations: int = 25, stream: bool = False,
                 history: Optional[List[Dict]] = None, enable_rich_debug: bool = True):
        """Initialize Agent with any LLM and optional tools.
        
        Args:
            llm: Any BaseLLM instance (vLLM, Gemini, Ollama, etc.)
            tools: List of callable functions to use as tools
            max_iterations: Maximum number of tool-calling iterations
            stream: If True, yields intermediate results during execution
            history: Previous conversation messages to continue from
            enable_rich_debug: If True, prints rich debug panels during execution
        """
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.stream = stream
        self.history = history or []  # Store conversation history
        self.enable_rich_debug = enable_rich_debug
        self._tool_functions = {tool.__name__: tool for tool in self.tools}
        self._conversation = []  # Current conversation (gets reset on new calls)
        self._console = Console()
    
    def _is_async_llm(self) -> bool:
        """Check if the LLM is async by checking if it has async __call__ method."""
        return asyncio.iscoroutinefunction(getattr(self.llm, '__call__', None))
    
    def _run_sync(self, input: Union[str, List[Dict]], **kwargs) -> Union[Dict, Generator]:
        """Execute agentic behavior synchronously for sync LLMs."""
        if self.stream or kwargs.get('stream', False):
            return self._stream_execute_sync(input, **kwargs)
        else:
            return self._execute_sync(input, **kwargs)
    
    async def run_async(self, input: Union[str, List[Dict]], **kwargs) -> Union[Dict, Generator]:
        """Execute agentic behavior - can be used exactly like an LLM.
        
        Args:
            input: User input as string or message list
            enable_rich_debug: Override instance setting for rich debug output
            **kwargs: Additional arguments passed to LLM
        
        Returns final result dict or Generator if stream=True.
        """
        if self.stream or kwargs.get('stream', False):
            return self._stream_execute_async(input, **kwargs)
        else:
            return await self._execute_async(input, **kwargs)

    def __call__(self, input: Union[str, List[Dict]], **kwargs):
        """Universal call that works as both sync and async."""
        return _AgentCall(self, input, **kwargs)
    
    async def acall(self, input: Union[str, List[Dict]], **kwargs):
        """Explicit async call - works with both sync and async LLMs."""
        if self._is_async_llm():
            return await self.run_async(input=input, **kwargs)
        else:
            # For sync LLMs in async context, run in thread pool
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, lambda: self._run_sync(input, **kwargs))

    
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
    
    async def _execute_async(self, input: Union[str, List[Dict]], **kwargs) -> Dict:
        """Non-streaming execution for async LLMs - returns final result."""
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
            response = await self.llm(messages, **kwargs)

            # Debug: Show assistant response
            if enable_debug:
                content = response.get('content', '')
                think = response.get('think', '')
                debug_text = ""
                if think:
                    debug_text += f"Think: {think}\n\n"
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
                
                tool_result = await self._execute_tool_async(tool_call_func)
                
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
        """Streaming execution for sync LLMs - yields intermediate results."""
        messages = self._normalize_input(input)
        # Prepend history to current conversation
        messages = self.history + messages
        self._conversation = messages.copy()
        
        if self.tools:
            kwargs['tools'] = self.tools
        
        for iteration in range(self.max_iterations):
            # Yield iteration start
            yield {
                'type': 'iteration_start',
                'iteration': iteration + 1,
                'messages_count': len(messages)
            }
            
            response = self.llm(messages, **kwargs)
            
            # Yield LLM response
            yield {
                'type': 'llm_response',
                'content': response.get('content', ''),
                'think': response.get('think', ''),
                'tool_calls': response.get('tool_calls', []),
                'iteration': iteration + 1
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
                    'messages': messages
                }
                return
            
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "tool_calls": response['tool_calls']
            })
            
            # Execute tools and yield results
            for i, tool_call in enumerate(response['tool_calls']):
                tool_func = tool_call['function']
                yield {
                    'type': 'tool_start',
                    'tool_name': tool_func['name'],
                    'tool_args': tool_func['arguments'],
                    'iteration': iteration + 1
                }
                
                tool_result = self._execute_tool_sync(tool_func)
                
                yield {
                    'type': 'tool_result',
                    'tool_name': tool_func['name'], 
                    'tool_args': tool_func['arguments'],
                    'result': str(tool_result),
                    'iteration': iteration + 1
                }
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get('id', f"call_{i}"),
                    "name": tool_func['name'],
                    "content": str(tool_result)
                })
        
        # Update conversation history
        self._conversation = messages
        
        # Max iterations reached
        yield {
            'type': 'final',
            'content': 'Max iterations reached',
            'iterations': self.max_iterations,
            'messages': messages,
            'truncated': True
        }
    
    async def _stream_execute_async(self, input: Union[str, List[Dict]], **kwargs) -> Generator[Dict, None, None]:
        """Streaming execution for async LLMs - yields intermediate results."""
        messages = self._normalize_input(input)
        # Prepend history to current conversation
        messages = self.history + messages
        self._conversation = messages.copy()
        
        if self.tools:
            kwargs['tools'] = self.tools
        
        for iteration in range(self.max_iterations):
            # Yield iteration start
            yield {
                'type': 'iteration_start',
                'iteration': iteration + 1,
                'messages_count': len(messages)
            }
            
            response = await self.llm(messages, **kwargs)
            
            # Yield LLM response
            yield {
                'type': 'llm_response',
                'content': response.get('content', ''),
                'think': response.get('think', ''),
                'tool_calls': response.get('tool_calls', []),
                'iteration': iteration + 1
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
                    'messages': messages
                }
                return
            
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "tool_calls": response['tool_calls']
            })
            
            # Execute tools and yield results
            for i, tool_call in enumerate(response['tool_calls']):
                tool_func = tool_call['function']
                yield {
                    'type': 'tool_start',
                    'tool_name': tool_func['name'],
                    'tool_args': tool_func['arguments'],
                    'iteration': iteration + 1
                }
                
                tool_result = await self._execute_tool_async(tool_func)
                
                yield {
                    'type': 'tool_result',
                    'tool_name': tool_func['name'], 
                    'tool_args': tool_func['arguments'],
                    'result': str(tool_result),
                    'iteration': iteration + 1
                }
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get('id', f"call_{i}"),
                    "name": tool_func['name'],
                    "content": str(tool_result)
                })
        
        # Update conversation history
        self._conversation = messages
        
        # Max iterations reached
        yield {
            'type': 'final',
            'content': 'Max iterations reached',
            'iterations': self.max_iterations,
            'messages': messages,
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
        tool_args = json.loads(str(tool_call['arguments']))

        if tool_name not in self._tool_functions:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            return self._tool_functions[tool_name](**tool_args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    async def _execute_tool_async(self, tool_call: Dict) -> Any:
        """Execute a single tool call asynchronously."""
        tool_name = tool_call['name']
        tool_args = json.loads(str(tool_call['arguments']))

        if tool_name not in self._tool_functions:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            if asyncio.iscoroutinefunction(self._tool_functions[tool_name]):
                return await self._tool_functions[tool_name](**tool_args)
            else:
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
        """Execute the agent chain sequentially."""
        current_input = input
        results = []
        
        for i, agent in enumerate(self.agents):
            # For first agent, use original input
            if i == 0:
                result = agent(current_input, **kwargs)
            else:
                # For subsequent agents, use previous agent's output as input
                current_input = result.get('content', '')
                result = agent(current_input, **kwargs)
            
            # Convert _AgentCall to dict result for chaining
            if isinstance(result, _AgentCall):
                result = result._execute_sync()
            
            results.append(result)
            
            # If streaming was requested, we can't chain properly
            # But _AgentCall objects are fine - they're lazy execution wrappers
            if (hasattr(result, '__iter__') and 
                not isinstance(result, (str, dict, _AgentCall)) and
                not hasattr(result, 'get')):  # Also allow dict-like objects
                raise ValueError("Cannot chain streaming agents. Set stream=False.")
        
        # Return final result with chain metadata
        final_result = results[-1].copy()
        final_result['chain_results'] = results
        final_result['chain_length'] = len(self.agents)
        
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
    
    # 5. Streaming with new features
    print("5. Streaming agent:")
    stream_agent = Agent(llm=llm, tools=[get_weather], stream=True)
    
    import asyncio
    async def demo_streaming():
        async for event in stream_agent.run_async("Get weather for London"):
            if event['type'] == 'tool_start':
                print(f"🔧 {event['tool_name']}")
            elif event['type'] == 'final':
                print(f"🏁 {event['content'][:50]}...")
                break
    
    asyncio.run(demo_streaming())
    
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
    print("   • Works with any BaseLLM, streaming, pluggable tools")
