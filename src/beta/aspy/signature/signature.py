# aspy/signature.py
from __future__ import annotations

from typing import Any, List, Optional, Union, get_args, get_origin
from pydantic import BaseModel, create_model, Field
import inspect


_PRIMS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "any": Any,
    "": str,  # default
}

# Global type registry for custom Pydantic models
_TYPE_REGISTRY: dict[str, type] = {}

def register_type(name: str, type_class: type) -> None:
    """Register a custom type for use in signatures."""
    if not (inspect.isclass(type_class) and issubclass(type_class, BaseModel)):
        raise ValueError(f"Type {name} must be a Pydantic BaseModel subclass")
    _TYPE_REGISTRY[name] = type_class

def get_registered_types() -> dict[str, type]:
    """Get all registered custom types."""
    return _TYPE_REGISTRY.copy()


def _is_optional(tp) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _field_def(py_type):
    """Return a (annotation, Field(default)) pair for create_model."""
    default = None if _is_optional(py_type) else ...
    return (py_type, Field(default))


def _parse_type(tok: str, custom_types: dict[str, type] = None):
    tok = tok.strip()
    # optional marker
    is_opt = tok.endswith("?")
    if is_opt:
        tok = tok[:-1].strip()

    t: Any
    # list: [T]
    if tok.startswith("[") and tok.endswith("]"):
        inner = _parse_type(tok[1:-1], custom_types)
        t = List[inner]  # typing List
    # custom types first, then primitives, then global registry, then caller's frame detection
    else:
        if custom_types and tok in custom_types:
            t = custom_types[tok]
        elif tok in _TYPE_REGISTRY:
            t = _TYPE_REGISTRY[tok]
        elif tok in _PRIMS:
            t = _PRIMS[tok]
        else:
            # Try to find the type in caller frames (DSPy-style detection)
            found_type = _detect_type_from_caller(tok)
            t = found_type if found_type else str

    if is_opt:
        t = Optional[t]
    return t

def _detect_type_from_caller(type_name: str):
    """Detect custom types from caller frames, similar to DSPy's approach."""
    frame = None
    try:
        frame = inspect.currentframe().f_back  # Start one level up
        max_frames = 50
        frame_count = 0

        while frame and frame_count < max_frames:
            frame_count += 1

            # Check locals first, then globals
            if type_name in frame.f_locals:
                potential_type = frame.f_locals[type_name]
                if (inspect.isclass(potential_type) and
                    issubclass(potential_type, BaseModel)):
                    return potential_type

            if frame.f_globals and type_name in frame.f_globals:
                potential_type = frame.f_globals[type_name]
                if (inspect.isclass(potential_type) and
                    issubclass(potential_type, BaseModel)):
                    return potential_type

            frame = frame.f_back

    except (AttributeError, ValueError):
        # Handle environments where frame introspection is not available
        pass
    finally:
        if frame:
            del frame

    return None


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
            # If no explicit type, try to detect if the field name itself is a custom type
            if not typ:
                detected_type = _detect_type_from_caller(name)
                if detected_type:
                    # Field name is actually the type name, use it as both
                    typ = name
        fields[name] = _parse_type(typ, custom_types)
    return fields


class Signature:
    """
    Minimal DSPy-like signature compiler.

    Examples:
        Signature("question -> answer")
        Signature("query, k:int? -> summary:str, sources:[str]")
        Signature("a:int, flags:[bool] -> ok:bool, result:str?")
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

        # Build Pydantic models dynamically
        in_fields = {k: _field_def(t) for k, t in self.input_types.items()}

        # Check if output is a single custom Pydantic model
        self.is_single_custom_output = False
        if len(self.output_types) == 1:
            output_type = list(self.output_types.values())[0]
            # Check if it's a custom BaseModel (not a primitive type)
            if (inspect.isclass(output_type) and
                issubclass(output_type, BaseModel) and
                output_type not in _PRIMS.values()):
                self.is_single_custom_output = True

        if self.is_single_custom_output:
            # For single custom model output, store the model class directly
            self.output_model_class = list(self.output_types.values())[0]
            self.Prediction = None  # No wrapper needed
        else:
            # For primitive types or multiple outputs, create Predict model
            out_fields = {k: _field_def(t) for k, t in self.output_types.items()}
            self.Prediction = create_model("Predict", __base__=BaseModel, **out_fields) # type: ignore
            self.output_model_class = None

        self.Input = create_model("Input", __base__=BaseModel, **in_fields)   # type: ignore

    def __call__(self):
        """Convenience: return (Input, Output) tuple."""
        if self.is_single_custom_output:
            return self.Input, self.output_model_class
        else:
            return self.Input, self.Prediction

    def __repr__(self):
        return f"Signature({self.spec!r})"
