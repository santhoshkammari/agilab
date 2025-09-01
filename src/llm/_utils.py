from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import Callable, Union

import pydantic


import contextlib
import json
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import (
  BaseModel,
  ByteSize,
  ConfigDict,
  Field,
  model_serializer,
)
from typing_extensions import Annotated, Literal


class SubscriptableBaseModel(BaseModel):
  def __getitem__(self, key: str) -> Any:
    if key in self:
      return getattr(self, key)

    raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    if key in self.model_fields_set:
      return True

    if value := self.__class__.model_fields.get(key):
      return value.default is not None

    return False

  def get(self, key: str, default: Any = None) -> Any:
    return getattr(self, key) if hasattr(self, key) else default


class Tool(SubscriptableBaseModel):
  type: Optional[str] = 'function'

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
    type='function',
    function=Tool.Function(
      name=func.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    ),
  )

  return Tool.model_validate(tool).model_dump()

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


if __name__ == '__main__':
    res = convert_function_to_tool(add_two_numbers)
    print(res)