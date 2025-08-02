from __future__ import annotations
import json
import inspect
import pydantic
import re
from abc import abstractmethod, ABC
from openai import OpenAI
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, Union, Callable, get_args, get_origin
from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
)
from typing_extensions import Literal


class SubscriptableBaseModel(BaseModel):
  def __getitem__(self, key: str) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg['role']
    'user'
    >>> msg = Message(role='user')
    >>> msg['nonexistent']
    Traceback (most recent call last):
    KeyError: 'nonexistent'
    """
    if key in self:
      return getattr(self, key)

    raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    """
    >>> msg = Message(role='user')
    >>> msg['role'] = 'assistant'
    >>> msg['role']
    'assistant'
    >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
    >>> msg = Message(role='user', content='hello')
    >>> msg['tool_calls'] = [tool_call]
    >>> msg['tool_calls'][0]['function']['name']
    'foo'
    """
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    """
    >>> msg = Message(role='user')
    >>> 'nonexistent' in msg
    False
    >>> 'role' in msg
    True
    >>> 'content' in msg
    False
    >>> msg.content = 'hello!'
    >>> 'content' in msg
    True
    >>> msg = Message(role='user', content='hello!')
    >>> 'content' in msg
    True
    >>> 'tool_calls' in msg
    False
    >>> msg['tool_calls'] = []
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = None
    >>> 'tool_calls' in msg
    True
    >>> tool = Tool()
    >>> 'type' in tool
    True
    """
    if key in self.model_fields_set:
      return True

    if value := self.model_fields.get(key):
      return value.default is not None

    return False

  def get(self, key: str, default: Any = None) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg.get('role')
    'user'
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent')
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent', 'default')
    'default'
    >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
    >>> msg.get('tool_calls')[0]['function']['name']
    'foo'
    """
    return getattr(self, key) if hasattr(self, key) else default




class Tool(SubscriptableBaseModel):
  type: Optional[Literal['function']] = 'function'

  class Function(SubscriptableBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    class Parameters(SubscriptableBaseModel):
      model_config = ConfigDict(populate_by_name=True)
      type: Optional[Literal['object']] = 'object'
      defs: Optional[Any] = Field(None, alias='$defs')
      items: Optional[Any] = None
      required: Optional[Sequence[str]] = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: Optional[Union[str, Sequence[str]]] = None
        items: Optional[Any] = None
        description: Optional[str] = None
        enum: Optional[Sequence[Any]] = None

      properties: Optional[Mapping[str, Property]] = None

    parameters: Optional[Parameters] = None

  function: Optional[Function] = None

def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  key = str(hash(doc_string))
  for line in doc_string.splitlines():
    lowered_line = line.lower().strip()
    if lowered_line.startswith('args:'):
      key = 'args'
    elif lowered_line.startswith(('returns:', 'yields:', 'raises:')):
      key = '_'

    else:
      # maybe change to a list and join later
      parsed_docstring[key] += f'{line.strip()}\n'

  last_key = None
  for line in parsed_docstring['args'].splitlines():
    line = line.strip()
    if ':' in line:
      # Split the line on either:
      # 1. A parenthetical expression like (integer) - captured in group 1
      # 2. A colon :
      # Followed by optional whitespace. Only split on first occurrence.
      parts = re.split(r'(?:\(([^)]*)\)|:)\s*', line, maxsplit=1)

      arg_name = parts[0].strip()
      last_key = arg_name

      # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
      arg_description = parts[-1].strip()
      if len(parts) > 2 and parts[1]:  # Has parenthetical content
        arg_description = parts[-1].split(':', 1)[-1].strip()

      parsed_docstring[last_key] = arg_description

    elif last_key and line:
      parsed_docstring[last_key] += ' ' + line

  return parsed_docstring


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string_hash = str(hash(inspect.getdoc(func)))
  parsed_docstring = _parse_docstring(inspect.getdoc(func))
  schema = type(
    func.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in inspect.signature(func).parameters.items()},
      '__signature__': inspect.signature(func),
      '__doc__': parsed_docstring[doc_string_hash],
    },
  ).model_json_schema()

  for k, v in schema.get('properties', {}).items():
    # If type is missing, the default is string
    types = {t.get('type', 'string') for t in v.get('anyOf')} if 'anyOf' in v else {v.get('type', 'string')}
    if 'null' in types:
      schema['required'].remove(k)
      types.discard('null')

    schema['properties'][k] = {
      'description': parsed_docstring[k],
      'type': ', '.join(types),
    }

  tool = Tool(
    function=Tool.Function(
      name=func.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    )
  )

  return Tool.model_validate(tool)


# Verifiers-style tool conversion for better compatibility
_JSON_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "integer", 
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

def _get_json_type(annotation: Any) -> tuple[str, list[Any] | None]:
    """Return the JSON Schema type name and optional enum values for annotation."""
    origin = get_origin(annotation)

    if origin is Literal:
        literal_values = list(get_args(annotation))
        if not literal_values:
            return "string", None
        first_value = literal_values[0]
        json_type = _JSON_PRIMITIVE_MAP.get(type(first_value), "string")
        return json_type, literal_values

    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _get_json_type(args[0])

    json_type = _JSON_PRIMITIVE_MAP.get(annotation, "string")
    return json_type, None

def _parse_docstring_verifiers(func: Any) -> tuple[str, dict[str, str]]:
    """Extract summary and parameter descriptions from function's docstring."""
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}

    lines = doc.splitlines()
    summary = next((line.strip() for line in lines if line.strip()), "")

    param_descs: dict[str, str] = {}
    try:
        block_idx = next(
            i for i, line in enumerate(lines)
            if line.strip().lower() in {"args:", "arguments:", "parameters:"}
        )
    except StopIteration:
        return summary, param_descs

    _PARAM_RE = re.compile(r"^\s*(\w+)\s*\(([^)]*)\):\s*(.*)$")
    for raw in lines[block_idx + 1:]:
        if not raw.strip():
            break
        match = _PARAM_RE.match(raw)
        if match:
            name, _type, desc = match.groups()
            param_descs[name] = desc.strip()
        else:
            if param_descs and raw.startswith(" " * 4):
                last_key = next(reversed(param_descs))
                param_descs[last_key] += " " + raw.strip()
            else:
                break
    return summary, param_descs

def _is_required(annotation: Any) -> bool:
    """True if annotation is not Optional/Union[..., None]."""
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) not in get_args(annotation)
    return True

def convert_func_to_oai_tool(func: Any) -> dict:
    """Convert function to OpenAI function-calling tool schema (Verifiers-style)."""
    if not callable(func):
        raise TypeError("Expected a callable object")

    signature = inspect.signature(func)
    summary, param_descs = _parse_docstring_verifiers(func)

    if not summary:
        summary = f"Auto-generated description for `{func.__name__}`."

    try:
        resolved_hints = inspect.get_annotations(func, eval_str=True)
    except AttributeError:
        from typing import get_type_hints
        resolved_hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name == "self":
            continue

        annotation = resolved_hints.get(
            name,
            param.annotation if param.annotation is not inspect.Parameter.empty else str,
        )
        json_type, enum_vals = _get_json_type(annotation)

        prop_schema: dict[str, Any] = {"type": json_type}
        if enum_vals is not None:
            prop_schema["enum"] = enum_vals

        if name in param_descs:
            prop_schema["description"] = param_descs[name]
        else:
            prop_schema.setdefault("description", f"Parameter `{name}` of type {json_type}.")

        properties[name] = prop_schema

        if param.default is inspect.Parameter.empty and _is_required(annotation):
            required.append(name)

    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": summary,
            "parameters": parameters_schema,
        },
    }


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model=None, tools=None, format=None, timeout=None):
        self._model = model
        self._tools = tools
        self._format = format
        self._timeout = timeout

    @abstractmethod
    def chat(self, input, **kwargs) -> dict:
        """Generate text using the LLM. Must be implemented by subclasses."""
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
                raise ValueError(f"Unsupported input type in list: {type(input[0]).__name__}. Expected str, list, or dict.")
        
        return self.chat(input=input, **kwargs)
    
    def batch_call(self, inputs, max_workers=4, **kwargs):
        """Process multiple inputs in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda text: self(text, **kwargs), inputs))
        return results

    def _normalize_input(self, input):
        """Convert string input to message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        return input

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
        """Get format from kwargs or use default format."""
        return kwargs.get('format', None) or self._format

    def _get_timeout(self, kwargs):
        """Get timeout from kwargs or use default timeout."""
        return kwargs.get('timeout', None) or self._timeout

    def _convert_function_to_tools(self, func: Optional[list[Callable]]) -> list[dict]:
        """Convert functions to OpenAI tools format using Verifiers-style conversion."""
        if not func:
            return []
        return [convert_func_to_oai_tool(f) if not isinstance(f, dict) else f for f in func]

    def _extract_thinking(self, content):
        """Extract thinking content from <think> tags."""
        import re
        think = ''
        if '<think>' in content and '</think>' in content:
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                think = think_match.group(1).strip()
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return think, content

    def _convert_tool_calls_to_dict(self, tool_calls):
        """Convert tool calls from object format to dictionary format."""
        if not tool_calls:
            return []
        
        dict_tool_calls = []
        for tool_call in tool_calls:
            if hasattr(tool_call, 'id'):
                # OpenAI format - convert to dict
                dict_tool_call = {
                    'id': tool_call.id,
                    'type': getattr(tool_call, 'type', 'function'),
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    }
                }
            else:
                # Already dict format
                dict_tool_call = tool_call
            
            dict_tool_calls.append(dict_tool_call)
        
        return dict_tool_calls


class vLLM(BaseLLM):
    def __init__(self, host: str = "0.0.0.0", port="8000", model=None, tools=None, format=None, timeout=None, api_key="dummy"):
        super().__init__(model=model, tools=tools, format=format, timeout=timeout)
        self._client = OpenAI(api_key=api_key, base_url=f"http://{host}:{port}/v1")

    def chat(self, input, **kwargs) -> dict:
        extra_body = {}
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)

        schema = self._get_format(kwargs)
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        if tools:
            tools = self._convert_function_to_tools(tools)

        if schema:
            extra_body['guided_json'] = schema
            if 'format' in kwargs:
                kwargs.pop('format')

        response = self._client.chat.completions.create(
            model=model,
            messages=input,
            extra_body=extra_body,
            tools=tools,
            timeout=timeout,
            **kwargs
        )

        result = {"think": "", "content": "", "tool_calls": None}

        result['content'] = response.choices[0].message.content or ""
        result['think'] = getattr(response.choices[0].message, 'reasoning_content', "") or ""

        # Convert tool_calls to dictionary format
        tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
        if tool_calls:
            result['tool_calls'] = self._convert_tool_calls_to_dict(tool_calls)
        else:
            result['tool_calls'] = []

        return result


class Ollama(BaseLLM):
    def __init__(self, host: str = "localhost", port: str = "11434", model=None, tools=None, format=None, timeout=None):
        super().__init__(model=model, tools=tools, format=format, timeout=timeout)
        from ollama import Client
        self._client = Client(host=f"http://{host}:{port}", timeout=timeout)

    def chat(self, input, **kwargs) -> dict:
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)
        tools = self._get_tools(kwargs)
        schema = self._get_format(kwargs)

        if tools:
            kwargs['tools'] = self._convert_function_to_tools(tools)
        if schema:
            kwargs['format'] = schema

        response = self._client.chat(model=model, messages=input, **kwargs)
        think, content = self._extract_thinking(response['message']['content'])

        # Convert tool_calls to dictionary format
        tool_calls = response.get('message', {}).get('tool_calls', None)
        converted_tool_calls = self._convert_tool_calls_to_dict(tool_calls) if tool_calls else []

        return {
            "think": think,
            "content": content,
            "tool_calls": converted_tool_calls
        }


class Person(BaseModel):
    name: str
    age: int

class CarInfo(BaseModel):
    brand: str
    model: str
    year: int

def get_weather(location: str) -> str:
    """Get current weather for a location"""
    return f"Weather in {location}: Sunny, 25°C"

# # Agentic loop - keep calling tools until no more tool calls
# while True:
#     response = llm(messages)
#
#     # Check if there are tool calls
#     if 'tools' not in response or not response['tools']:
#         # No more tool calls, show final content
#         print("=== FINAL RESPONSE ===")
#         print(response.get('content', 'No content'))
#         break
#
#     # Add the assistant message with tool calls first
#     messages.append({
#         "role": "assistant",
#         "tool_calls": [
#             {
#                 "id": tool_call.get('id', f"call_{i}"),
#                 "function": {
#                     "name": tool_call['name'],
#                     "arguments": json.dumps(tool_call['arguments'])
#                 },
#                 "type": "function"
#             }
#             for i, tool_call in enumerate(response['tools'])
#         ]
#     })
#
#     # Process each tool call and add tool results
#     for i, tool_call in enumerate(response['tools']):
#         tool_name = tool_call['name']
#         tool_args = tool_call['arguments']
#         tool_id = tool_call.get('id', f"call_{i}")
#
#         print(f"Calling tool: {tool_name} with args: {tool_args}")
#
#         # Execute the tool
#         tool_result = tool_functions[tool_name](**tool_args)
#
#         # Add tool result back to messages
#         messages.append({
#             "role": "tool",
#             "tool_call_id": tool_id,
#             "name": tool_name,
#             "content": str(tool_result)
#         })

if __name__ == '__main__':
    print("=== LLM Framework Examples ===\n")
    
    # Example 1: Basic usage with __call__
    print("1. Basic text generation:")
    llm = vLLM(host='192.168.170.76', port="8077", 
               model="/home/ng6309/datascience/santhosh/models/Qwen3-14B")
    res = llm("What is 2+3?")
    print(f"Answer: {res['content']}\n")
    
    # # Example 2: JSON schema formatting
    # print("2. Structured JSON output:")
    # llm_json = vLLM(host='192.168.170.76', port="8077",
    #                 model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
    #                 format=Person.model_json_schema())
    # res = llm_json("Generate a person named John who is 30 years old")
    # print(f"JSON: {res['content']}\n")
    
    # Example 3: Tool usage
    print("3. Tool usage:")
    llm_tools = vLLM(host='192.168.170.76', port="8077",
                     model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
                     tools=[get_weather])
    res = llm_tools("What's the weather in Paris?")
    print(f"Tools called: {res['tools']}\n")
    
    # Example 4: Runtime overrides
    print("4. Runtime parameter override:")
    base_llm = vLLM(host='192.168.170.76', port="8077",
                    model="/home/ng6309/datascience/santhosh/models/Qwen3-14B")
    res = base_llm("Generate car info", 
                   format=CarInfo.model_json_schema(),
                   timeout=10)
    print(f"Car info: {res['content']}\n")
    
    # Example 5: Ollama supports only with timeout in init
    print("5. Ollama with init timeout:")
    # ollama_llm = Ollama(host='192.168.170.76', port="11434", 
    #                     model="qwen3:0.6b-fp16", timeout=30)
    # res = ollama_llm("Hello world")
    # print(f"Response: {res['content']}\n")
    
    # Example 6: Batch processing with automatic threading
    print("6. Batch processing with batch_call:")
    batch_llm = vLLM(host='192.168.170.76', port="8077",
                     model="/home/ng6309/datascience/santhosh/models/Qwen3-14B")
    texts = ['What is 2+2?', 'What is 3+3?']
    batch_results = batch_llm.batch_call(texts, max_workers=2)
    print(f"Batch outputs: {[r['content'] for r in batch_results]}\n")
    
    print("=== Use Cases ===")
    print("• Data generation: llm(prompt, format=schema)")
    print("• Agent systems: llm(prompt, tools=[func1, func2])")
    print("• Inference: llm(text) or dataset.map(llm)")
    print("• Batch processing: llm.batch_call(texts, max_workers=N)")
    print("• Synthetic data: llm(template) with different formats")
    print("• Flexible timeouts: init or runtime override")


    # Case 1: Single string
    print("1. Single string:")
    res = base_llm("What is 2+2?", temperature=0)
    print(f"Result: {res['content']}\n")

    # Case 2: List of strings (batch)
    print("2. List of strings (batch):")
    batch_strings = ["What is 2+2?", "What is 3+3?", "What is 5+5?"]
    res = base_llm(batch_strings, temperature=0)
    print(f"Results: {[r['content'] for r in res]}\n")

    # Case 3: Single conversation (list of dicts)
    print("3. Single conversation (list of dicts):")
    single_conv = [{"role": "user", "content": "Hi, just tell me hello"}]
    res = base_llm(single_conv, temperature=0)
    print(f"Result: {res['content']}\n")

    # Case 4: Multi-turn conversation
    print("4. Multi-turn conversation:")
    multi_conv = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I'm Claude"},
        {"role": "user", "content": "Nice to meet you!"}
    ]
    res = base_llm(multi_conv, temperature=0)
    print(f"Result: {res['content']}\n")

    # Case 5: List of lists (batch conversations)
    print("5. List of lists (batch conversations):")
    batch_convs = [
        [{"role": "user", "content": "Say hi"}],
        [{"role": "user", "content": "Say bye"}],
        [{"role": "user", "content": "Say thanks"}]
    ]
    res = base_llm(batch_convs, temperature=0)
    print(f"Results: {[r['content'] for r in res]}\n")


