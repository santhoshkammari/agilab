# aspy/signature.py
from __future__ import annotations

from typing import Any, List, Optional, Union, get_args, get_origin
from pydantic import BaseModel, create_model, Field


_PRIMS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "any": Any,
    "": str,  # default
}


def _is_optional(tp) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _field_def(py_type):
    """Return a (annotation, Field(default)) pair for create_model."""
    default = None if _is_optional(py_type) else ...
    return (py_type, Field(default))


def _parse_type(tok: str):
    tok = tok.strip()
    # optional marker
    is_opt = tok.endswith("?")
    if is_opt:
        tok = tok[:-1].strip()

    t: Any
    # list: [T]
    if tok.startswith("[") and tok.endswith("]"):
        inner = _parse_type(tok[1:-1])
        t = List[inner]  # typing List
    # primitive or default
    else:
        t = _PRIMS.get(tok, str)

    if is_opt:
        t = Optional[t]
    return t


def _parse_fields(side: str) -> dict[str, Any]:
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
        fields[name] = _parse_type(typ)
    return fields


class Signature:
    """
    Minimal DSPy-like signature compiler.

    Examples:
        Signature("question -> answer")
        Signature("query, k:int? -> summary:str, sources:[str]")
        Signature("a:int, flags:[bool] -> ok:bool, result:str?")
    """

    def __init__(self, spec: str):
        self.spec = spec.strip()
        if "->" not in self.spec:
            raise ValueError("Signature must contain '->' (e.g., 'question -> answer').")
        left, right = map(str.strip, self.spec.split("->", 1))

        self.input_types = _parse_fields(left)
        self.output_types = _parse_fields(right)

        # Build Pydantic models dynamically
        in_fields = {k: _field_def(t) for k, t in self.input_types.items()}
        out_fields = {k: _field_def(t) for k, t in self.output_types.items()}

        self.Input = create_model("Input", __base__=BaseModel, **in_fields)   # type: ignore
        self.Prediction = create_model("Prediction", __base__=BaseModel, **out_fields) # type: ignore

    def __call__(self):
        """Convenience: return (Input, Prediction) tuple."""
        return self.Input, self.Prediction

    def __repr__(self):
        return f"Signature({self.spec!r})"
