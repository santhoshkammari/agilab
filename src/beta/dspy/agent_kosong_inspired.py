import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from pydantic import BaseModel
from lm import LM


# Tool return types
@dataclass
class ToolOk:
    """Successful tool execution result"""
    output: str
    message: str = ""
    brief: str = ""


@dataclass
class ToolError:
    """Tool execution error"""
    output: str = ""
    message: str = "Error occurred"
    brief: str = "Error"


ToolReturnType = Union[ToolOk, ToolError]


# Type-safe tool definition with Pydantic
class CallableTool(ABC):
    """Base class for type-safe tools with Pydantic parameters"""

    name: str
    description: str
    params: type[BaseModel]

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from Pydantic model"""
        schema = self.params.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema
            }
        }

    async def call(self, arguments: str) -> ToolReturnType:
        """Parse arguments and execute tool"""
        try:
            # Parse JSON arguments
            args_dict = json.loads(arguments)
            # Validate with Pydantic
            params = self.params.model_validate(args_dict)
            # Execute tool
            return await self.__call__(params)
        except json.JSONDecodeError as e:
            return ToolError(message=f"Invalid JSON: {e}", brief="Parse error")
        except Exception as e:
            return ToolError(message=str(e), brief="Validation error")

    @abstractmethod
    async def __call__(self, params: BaseModel) -> ToolReturnType:
        """Implement tool logic"""
        ...


# Simple toolset
class SimpleToolset:
    """Manages tool registration and execution"""

    def __init__(self):
        self.tools: Dict[str, CallableTool] = {}

    def add(self, tool: CallableTool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM"""
        return [tool.get_schema() for tool in self.tools.values()]

    async def execute(self, tool_name: str, arguments: str) -> ToolReturnType:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return ToolError(message=f"Tool {tool_name} not found", brief="Not found")
        return await self.tools[tool_name].call(arguments)


# Agent with kosong-inspired design
class Agent:
    def __init__(self, llm: LM = None):
        self.llm = llm or LM(
            model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            api_base="http://localhost:8000"
        )
        self.messages = []
        self.toolset = SimpleToolset()

    async def run(self, messages: List[Dict], tools: Optional[List[CallableTool]] = None):
        """
        Single agent step with tool support.
        Yields streaming chunks and executes tools concurrently.
        """
        # Add messages to history
        self.messages.extend(messages)

        # Register tools
        if tools:
            for tool in tools:
                self.toolset.add(tool)

        # Get tool schemas
        tool_schemas = self.toolset.get_schemas() if tools else None

        # Stream response
        full_content = ""
        tool_calls = []
        stream_params = {}
        if tool_schemas:
            stream_params['tool_choice'] = 'required'

        async for chunk in self.llm.stream(self.messages, tools=tool_schemas, **stream_params):
            yield chunk

            # Accumulate content and tool calls
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    full_content += delta['content']
                if 'tool_calls' in delta:
                    # Accumulate tool calls
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})

                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'type' in tc:
                            tool_calls[idx]['type'] = tc['type']
                        if 'function' in tc:
                            if tc['function'].get('name'):
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            if tc['function'].get('arguments'):
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

        # Save assistant message
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg['tool_calls'] = tool_calls
        self.messages.append(assistant_msg)

        # Execute tools concurrently
        if tool_calls:
            print(f"\n\nExecuting {len(tool_calls)} tool(s)...")
            tasks = []
            for tc in tool_calls:
                tool_name = tc['function']['name']
                arguments = tc['function']['arguments']
                tasks.append(self.toolset.execute(tool_name, arguments))

            # Execute all tools concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Add tool results to messages
            for tc, result in zip(tool_calls, results):
                if isinstance(result, Exception):
                    output = f"Error: {result}"
                elif isinstance(result, ToolOk):
                    output = result.output
                else:
                    output = result.message

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": output
                })
                print(f"Tool {tc['function']['name']}: {output}")


# Example tool using Pydantic
class WeatherParams(BaseModel):
    city: str
    unit: str = "celsius"


class GetWeatherTool(CallableTool):
    name = "get_weather"
    description = "Get the current weather for a city"
    params = WeatherParams

    async def __call__(self, params: WeatherParams) -> ToolReturnType:
        # Simulate API call
        return ToolOk(
            output=f"Weather in {params.city}: 22 degrees {params.unit}",
            brief=f"22°{params.unit[0].upper()}"
        )


class AddParams(BaseModel):
    a: int
    b: int


class AddTool(CallableTool):
    name = "add"
    description = "Add two integers"
    params = AddParams

    async def __call__(self, params: AddParams) -> ToolReturnType:
        result = params.a + params.b
        return ToolOk(output=str(result), brief=f"{result}")


async def main():
    agent = Agent()

    # Example 1: Simple text
    print("=== Example 1: Simple chat ===")
    messages = [{"role": "user", "content": "Say hello in one sentence."}]
    async for chunk in agent.run(messages):
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta:
                print(delta['content'], end='', flush=True)
    print("\n")

    # Example 2: Tool calling
    print("\n=== Example 2: Tool calling ===")
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    async for chunk in agent.run(messages, tools=[GetWeatherTool()]):
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta and delta['content']:
                print(delta['content'], end='', flush=True)
    print("\n")

    # Example 3: Multiple tools
    print("\n=== Example 3: Multiple tools ===")
    messages = [{"role": "user", "content": "Add 15 and 27"}]
    async for chunk in agent.run(messages, tools=[AddTool()]):
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta and delta['content']:
                print(delta['content'], end='', flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
