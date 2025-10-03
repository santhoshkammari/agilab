import requests
from transformers.utils import get_json_schema
from dataclasses import dataclass
from typing import Any, List, Optional, Union, get_args, get_origin
from pydantic import BaseModel, create_model, Field
import inspect
import json
import contextvars

# Global LM context
_current_lm = contextvars.ContextVar('lm', default=None)

def configure(lm=None, **kwargs):
    """Configure the framework with default LM."""
    if lm:
        _current_lm.set(lm)

def get_lm():
    """Get the current LM from context."""
    return _current_lm.get()

def _set_lm(lm):
    """Internal: Set the default LM."""
    _current_lm.set(lm)

@dataclass
class Prediction:
    raw: dict
    content: str
    tools: list
    prompt_tokens: int
    completion_tokens: int

    def __str__(self):
        return f"Prediction(content={self.content!r}, tools={self.tools}, prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens})"

class LM:
    def __init__(self, api_base="http://localhost:11434", model: str = ""):
        self.model = model
        self.api_base = api_base

    def __call__(self,**kwargs):
        if 'tools' in kwargs:
            kwargs['tools'] = [get_json_schema(f) if callable(f) else f for f in kwargs['tools']]
        return self.call_llm(**kwargs)

    def call_llm(self, messages, **params):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        stream = params.get('stream', False)
        response = requests.post(url=f"{self.api_base}/v1/chat/completions",
                                json={"model": self.model, "messages": messages, **params},
                                stream=stream)
        if stream:
            return response
        raw = response.json()
        message = raw['choices'][0]['message']
        content = message.get('content', '')
        tools = message.get('tool_calls', [])
        prompt_tokens = raw['usage']['prompt_tokens']
        completion_tokens = raw['usage']['completion_tokens']
        return Prediction(raw=raw, content=content, tools=tools,
                         prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


# ============ Signature System ============

_PRIMS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "any": Any,
    "": str,
}

_TYPE_REGISTRY: dict[str, type] = {}

def register_type(name: str, type_class: type) -> None:
    """Register a custom type for use in signatures."""
    if not (inspect.isclass(type_class) and issubclass(type_class, BaseModel)):
        raise ValueError(f"Type {name} must be a Pydantic BaseModel subclass")
    _TYPE_REGISTRY[name] = type_class

def _is_optional(tp) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)

def _field_def(py_type):
    default = None if _is_optional(py_type) else ...
    return (py_type, Field(default))

def _parse_type(tok: str, custom_types: dict[str, type] = None):
    tok = tok.strip()
    is_opt = tok.endswith("?")
    if is_opt:
        tok = tok[:-1].strip()

    t: Any
    if tok.startswith("[") and tok.endswith("]"):
        inner = _parse_type(tok[1:-1], custom_types)
        t = List[inner]
    else:
        if custom_types and tok in custom_types:
            t = custom_types[tok]
        elif tok in _TYPE_REGISTRY:
            t = _TYPE_REGISTRY[tok]
        elif tok in _PRIMS:
            t = _PRIMS[tok]
        else:
            t = str

    if is_opt:
        t = Optional[t]
    return t

def _parse_fields(side: str, custom_types: dict[str, type] = None) -> dict[str, Any]:
    side = side.strip()
    if not side:
        return {}
    fields: dict[str, Any] = {}
    for raw in side.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            name, typ = token.split(":", 1)
            name, typ = name.strip(), typ.strip()
        else:
            name, typ = token, ""
        fields[name] = _parse_type(typ, custom_types)
    return fields

class Signature:
    """Signature compiler for defining input -> output schemas.

    Examples:
        Signature("question -> answer")
        Signature("query, k:int -> summary:str")
    """
    def __init__(self, spec: str, instructions: str = None, custom_types: dict[str, type] = None):
        self.spec = spec.strip()
        self.instructions = instructions
        self.custom_types = custom_types or {}

        if "->" not in self.spec:
            raise ValueError("Signature must contain '->' (e.g., 'question -> answer').")
        left, right = map(str.strip, self.spec.split("->", 1))

        self.input_types = _parse_fields(left, self.custom_types)
        self.output_types = _parse_fields(right, self.custom_types)

        in_fields = {k: _field_def(t) for k, t in self.input_types.items()}
        out_fields = {k: _field_def(t) for k, t in self.output_types.items()}

        self.Input = create_model("Input", __base__=BaseModel, **in_fields)
        self.Output = create_model("Output", __base__=BaseModel, **out_fields)

    def __call__(self):
        return self.Input, self.Output

    def __repr__(self):
        return f"Signature({self.spec!r})"

class Predict:
    """Prediction module using Signature for structured input/output."""

    def __init__(self, signature: Union[str, Signature] = None, lm: LM = None, instructions: str = None):
        if signature:
            if isinstance(signature, str):
                self.signature = Signature(signature, instructions=instructions)
            else:
                self.signature = signature
        else:
            self.signature = None
        self.lm = lm

    def _build_prompt(self, **inputs):
        Input, Output = self.signature()
        input_instance = Input(**inputs)
        output_schema = Output.model_json_schema()

        system_parts = ["<system_prompt>"]
        system_parts.append("You are a helpful assistant that provides accurate responses.")
        if self.signature.instructions:
            system_parts.append(f"\nInstructions: {self.signature.instructions}")
        system_parts.append("</system_prompt>")
        system_parts.append("\n<output_format>")
        system_parts.append("Respond with valid JSON matching this schema:")
        system_parts.append(json.dumps(output_schema, indent=2))
        system_parts.append("</output_format>")

        user_parts = ["<input>"]
        for field_name, value in input_instance.model_dump().items():
            user_parts.append(f"{field_name}: {value}")
        user_parts.append("</input>")

        return [
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": "\n".join(user_parts)}
        ]

    def _get_response_format(self):
        _, Output = self.signature()
        schema = Output.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "Output",
                "schema": schema,
                "strict": True
            }
        }

    def __call__(self, **inputs):
        lm = self.lm or get_lm()
        if not lm:
            raise ValueError("No LM set. Use configure(lm=...) or pass lm in constructor.")

        if not self.signature:
            raise ValueError("No signature set. Pass signature in constructor.")

        messages = self._build_prompt(**inputs)
        response_format = self._get_response_format()

        # Use LM without tools, with response_format
        raw_response = lm.call_llm(messages, response_format=response_format)

        # Extract content and parse
        content = raw_response.content
        try:
            output_data = json.loads(content)
        except json.JSONDecodeError:
            output_fields = list(self.signature.output_types.keys())
            if len(output_fields) == 1:
                output_data = {output_fields[0]: content}
            else:
                output_data = {field: content for field in output_fields}

        # Return as dict
        _, Output = self.signature()
        try:
            prediction_instance = Output(**output_data)
            return prediction_instance
        except Exception:
            return output_data


# ============ Example Tools ============

def get_weather(location: str, unit: str = "celsius"):
    """Get the weather for a location.

    Args:
        location: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{unit[0].upper()}"

if __name__ == '__main__':
    lm = LM(api_base='http://192.168.170.76:8000')
    configure(lm=lm)

    response = lm(messages='what is weather in gurgaon', tools=[get_weather])
    print(response)

    predictor = Predict("question -> answer")
    result = predictor(question="What is the capital of France?")
    print(result)

    classifier = Predict("text -> sentiment:str, confidence:float")
    output = classifier(text="I love this product!")
    print(output)

    # lm = LM(api_base='http://192.168.170.76:8000')
    # response = lm(messages='what is weather in gurgaon /no_think', tools=[get_weather])
    # print(response)

    # lm = LM(api_base='http://192.168.170.76:8000')
    # response = lm(messages='what is weather in gurgaon', tools=[get_weather], stream=True)
    # for line in response.iter_lines():
    #     if line:
    #         print(line.decode('utf-8'))
