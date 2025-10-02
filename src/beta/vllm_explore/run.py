
import xgrammar as xgr
import json
from pydantic import BaseModel
from typing import Literal

class GetWeatherInput(BaseModel):
    """Input schema for get_weather function."""
    location: str 
    unit: Literal["celsius", "fahrenheit"]
    
grammar = xgr.Grammar.from_json_schema(json.dumps(GetWeatherInput.model_json_schema()))

print(f"\n✅ Grammar created: {type(grammar)}")
print(f"   Class: {grammar.__class__.__name__}")
print(f"   Module: {grammar.__class__.__module__}")

print('--------')
print(grammar.__str__())
