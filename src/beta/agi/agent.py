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
        self._last_messages = None

    @property
    def messages(self):
        """Get the last built messages."""
        return self._last_messages

    @property
    def prompt(self):
        """Get the last built prompt as a formatted string."""
        if not self._last_messages:
            return None
        parts = []
        for msg in self._last_messages:
            role = msg['role'].upper()
            content = msg['content']
            parts.append(f"=== {role} ===\n{content}\n")
        return "\n".join(parts)

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
        self._last_messages = messages
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


# ============ Evaluation System ============

class Example:
    """Container for evaluation examples."""

    def __init__(self, **kwargs):
        self._input_keys = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def with_inputs(self, *keys):
        """Set which fields should be treated as inputs."""
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        """Return input fields (excludes answer/output/target/label)."""
        all_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

        if self._input_keys is not None:
            return {k: v for k, v in all_attrs.items() if k in self._input_keys}
        else:
            target_fields = {'answer', 'output', 'target', 'label'}
            return {k: v for k, v in all_attrs.items() if k not in target_fields}

    def copy(self, **kwargs):
        """Create a copy of this Example."""
        data = self.model_dump()
        data.update(kwargs)
        new_example = Example(**data)
        new_example._input_keys = self._input_keys
        return new_example

    def model_dump(self):
        """Return all attributes as a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        attrs = self.model_dump()
        attr_str = ", ".join([f"{k}={repr(v)}" for k, v in attrs.items()])
        return f"Example({attr_str})"


class Evaluate:
    """Evaluation class with live saving and auto-resume support."""

    def __init__(self,
                 devset: List[Example],
                 metrics,
                 output_file: str = None,
                 display_progress: bool = True,
                 force_restart: bool = False):
        """
        Args:
            devset: List of Examples to evaluate
            metrics: Single callable OR list of callables OR dict {name: callable}
            output_file: Path to JSON file for live saving and resume
            display_progress: Show tqdm progress bar
            force_restart: Ignore existing results and start fresh
        """
        self.devset = devset
        self.metrics = self._normalize_metrics(metrics)
        self.output_file = output_file
        self.display_progress = display_progress
        self.force_restart = force_restart
        self.completed_indices = set()
        self.results = []

        if self.output_file and not self.force_restart:
            self._load_existing_results()

    def _normalize_metrics(self, metrics):
        """Normalize metrics input to dict format."""
        if callable(metrics):
            # Single callable -> dict with function name
            name = metrics.__name__ if hasattr(metrics, '__name__') else 'metric'
            return {name: metrics}
        elif isinstance(metrics, list):
            # List of callables -> dict with function names
            return {m.__name__ if hasattr(m, '__name__') else f'metric_{i}': m
                    for i, m in enumerate(metrics)}
        elif isinstance(metrics, dict):
            # Already a dict
            return metrics
        else:
            raise ValueError("metrics must be a callable, list of callables, or dict")

    def _load_existing_results(self):
        """Load existing results from JSON file if it exists."""
        try:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
                self.results = data.get('results', [])
                self.completed_indices = {r['example_id'] for r in self.results}
                if self.completed_indices:
                    print(f"Resuming from {self.output_file}: {len(self.completed_indices)} examples already completed")
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {self.output_file}, starting fresh")

    def _save_results(self, total_scores: dict, completed: int):
        """Save current results to JSON file."""
        if not self.output_file:
            return

        # Calculate final scores as percentages
        final_scores = {
            name: (total_scores[name] / len(self.devset)) * 100 if self.devset else 0.0
            for name in self.metrics.keys()
        }

        data = {
            "scores": final_scores,
            "total_examples": len(self.devset),
            "completed": completed,
            "results": self.results
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def __call__(self, module):
        """Run evaluation on the module."""
        # Initialize total scores for each metric
        total_scores = {name: sum(r['scores'].get(name, 0.0) for r in self.results)
                       for name in self.metrics.keys()}

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            print("Install tqdm for progress bars: pip install tqdm")

        # Create iterator
        if has_tqdm and self.display_progress:
            iterator = tqdm(enumerate(self.devset), total=len(self.devset), desc="Evaluating")
        else:
            iterator = enumerate(self.devset)

        for idx, example in iterator:
            # Skip already completed
            if idx in self.completed_indices:
                continue

            try:
                # Get prediction
                inputs = example.inputs()
                prediction = module(**inputs)

                # Calculate scores for all metrics
                scores = {}
                for metric_name, metric_func in self.metrics.items():
                    score = metric_func(example, prediction)
                    scores[metric_name] = score
                    total_scores[metric_name] += score

                # Store result
                result = {
                    "example_id": idx,
                    "inputs": inputs,
                    "prediction": prediction if isinstance(prediction, dict) else str(prediction),
                    "scores": scores
                }
                self.results.append(result)
                self.completed_indices.add(idx)

                # Live save
                self._save_results(total_scores, len(self.completed_indices))

                # Update progress
                if has_tqdm and self.display_progress:
                    current_avgs = {name: total_scores[name] / len(self.completed_indices)
                                   for name in self.metrics.keys()}
                    postfix_str = " ".join([f"{name}={avg:.3f}" for name, avg in current_avgs.items()])
                    iterator.set_postfix_str(postfix_str)

            except Exception as e:
                print(f"\nError evaluating example {idx}: {e}")
                result = {
                    "example_id": idx,
                    "inputs": example.inputs(),
                    "prediction": {"error": str(e)},
                    "scores": {name: 0.0 for name in self.metrics.keys()}
                }
                self.results.append(result)
                self.completed_indices.add(idx)
                self._save_results(total_scores, len(self.completed_indices))

        # Final scores
        final_scores = {
            name: (total_scores[name] / len(self.devset)) * 100 if self.devset else 0.0
            for name in self.metrics.keys()
        }

        print(f"\nEvaluation complete:")
        for name, score in final_scores.items():
            raw_score = total_scores[name]
            print(f"  {name}: {score:.2f}% ({raw_score:.1f}/{len(self.devset)})")

        return {"scores": final_scores, "results": self.results}


def exact_match(example: Example, prediction) -> float:
    """Exact match metric."""
    expected = getattr(example, 'answer', getattr(example, 'output', ''))
    if isinstance(prediction, dict):
        predicted = prediction.get('answer', prediction.get('output', ''))
    else:
        predicted = getattr(prediction, 'answer', getattr(prediction, 'output', ''))
    return 1.0 if str(expected).strip() == str(predicted).strip() else 0.0


def contains_match(example: Example, prediction) -> float:
    """Check if prediction contains the expected answer."""
    expected = str(getattr(example, 'answer', getattr(example, 'output', ''))).strip().lower()
    if isinstance(prediction, dict):
        predicted = str(prediction.get('answer', prediction.get('output', ''))).strip().lower()
    else:
        predicted = str(getattr(prediction, 'answer', getattr(prediction, 'output', ''))).strip().lower()
    return 1.0 if expected in predicted else 0.0


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

    devset = [
        Example(question="What is 2+2?", answer="4"),
        Example(question="What is 3+3?", answer="6"),
    ]*10
    qa_predictor = Predict("question -> answer")

    # Single metric
    evaluator = Evaluate(devset=devset, metrics=exact_match, output_file="eval_single.json")
    eval_result = evaluator(qa_predictor)
    print(eval_result)

    # Multiple metrics
    evaluator_multi = Evaluate(devset=devset, metrics=[exact_match, contains_match], output_file="eval_multi.json")
    eval_result_multi = evaluator_multi(qa_predictor)
    print(eval_result_multi)

    # lm = LM(api_base='http://192.168.170.76:8000')
    # response = lm(messages='what is weather in gurgaon /no_think', tools=[get_weather])
    # print(response)

    # lm = LM(api_base='http://192.168.170.76:8000')
    # response = lm(messages='what is weather in gurgaon', tools=[get_weather], stream=True)
    # for line in response.iter_lines():
    #     if line:
    #         print(line.decode('utf-8'))
