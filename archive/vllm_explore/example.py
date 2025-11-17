#!/usr/bin/env python3
"""
Example: Converting Functions and Pydantic Models to Grammar for vLLM
=====================================================================

This example demonstrates:
1. Function to JSON schema conversion (manual)
2. Pydantic model to JSON schema conversion (built-in)
3. Using schemas with vLLM's guided decoding
"""

import json
import inspect
from typing import get_type_hints, Literal, Optional
from pydantic import BaseModel, Field
from vllm.sampling_params import GuidedDecodingParams, SamplingParams


# ============================================================================
# APPROACH 1: Function to Schema (Manual Conversion)
# ============================================================================

def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    Get the current weather for a location.

    Args:
        location: The city name or location
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information dict
    """
    # This is just a placeholder function
    return {"temperature": 22, "conditions": "sunny"}


def function_to_json_schema(func) -> dict:
    """
    Convert a Python function to JSON schema.

    This extracts function signature and type hints to create a schema
    that can be used for structured output generation.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    # Type mapping from Python to JSON schema
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        json_type = type_mapping.get(param_type, "string")

        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter: {param_name}"
        }

        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }

    return schema


# ============================================================================
# APPROACH 2: Pydantic Model to Schema (Built-in, Recommended)
# ============================================================================

class GetWeatherInput(BaseModel):
    """Input schema for get_weather function."""

    location: str = Field(
        ...,  # Required field
        description="City name or location to get weather for",
        examples=["San Francisco", "New York", "Tokyo"]
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit for temperature values"
    )


class SearchQuery(BaseModel):
    """Input schema for search function."""

    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=10, description="Maximum number of results", ge=1, le=100)
    include_metadata: bool = Field(default=False, description="Include metadata in results")


class SendEmailInput(BaseModel):
    """Input schema for send_email function."""

    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    cc: Optional[list[str]] = Field(default=None, description="CC recipients")


# ============================================================================
# Main Example
# ============================================================================

def main():
    print("=" * 80)
    print("vLLM: Function & Pydantic Model to Grammar Conversion Examples")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Example 1: Function to Schema (Manual)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Function → Schema (Manual Conversion)")
    print("=" * 80)

    schema_from_function = function_to_json_schema(get_weather)
    print(f"\n📝 Function: {get_weather.__name__}")
    print(f"   Signature: {inspect.signature(get_weather)}")
    print(f"\n✅ Generated Schema:")
    print(json.dumps(schema_from_function, indent=2))

    # Create GuidedDecodingParams from function schema
    guided_params_func = GuidedDecodingParams.from_optional(json=schema_from_function)
    print(f"\n✅ GuidedDecodingParams created: {guided_params_func is not None}")

    # -------------------------------------------------------------------------
    # Example 2: Pydantic Model to Schema (Built-in)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Pydantic Model → Schema (Built-in, Recommended)")
    print("=" * 80)

    # Get schema from Pydantic model
    schema_from_pydantic = GetWeatherInput.model_json_schema()
    print(f"\n📝 Pydantic Model: {GetWeatherInput.__name__}")
    print(f"   Fields: {list(GetWeatherInput.model_fields.keys())}")
    print(f"\n✅ Generated Schema (richer with descriptions, defaults, enums):")
    print(json.dumps(schema_from_pydantic, indent=2))

    # Create GuidedDecodingParams from Pydantic model (EASIEST WAY!)
    guided_params_pydantic = GuidedDecodingParams.from_optional(json=GetWeatherInput)
    print(f"\n✅ GuidedDecodingParams created: {guided_params_pydantic is not None}")

    # -------------------------------------------------------------------------
    # Example 3: Using with vLLM SamplingParams
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Using with vLLM SamplingParams")
    print("=" * 80)

    # Different models for different use cases
    examples = [
        ("get_weather", GetWeatherInput),
        ("search", SearchQuery),
        ("send_email", SendEmailInput)
    ]

    for func_name, model_class in examples:
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic output
            max_tokens=200,
            guided_decoding=GuidedDecodingParams.from_optional(json=model_class)
        )

        print(f"\n✅ {func_name}:")
        print(f"   Model: {model_class.__name__}")
        print(f"   Temperature: {sampling_params.temperature}")
        print(f"   Has guided decoding: {sampling_params.guided_decoding is not None}")
        print(f"   Schema fields: {list(model_class.model_fields.keys())}")

    # -------------------------------------------------------------------------
    # Example 4: Validating Generated Outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Validating LLM Outputs Against Schema")
    print("=" * 80)

    # Simulate LLM outputs
    test_outputs = [
        {"location": "San Francisco", "unit": "celsius"},
        {"location": "New York", "unit": "fahrenheit"},
        {"location": "Tokyo"},  # Uses default unit
        {"location": "London", "unit": "kelvin"},  # Invalid unit!
    ]

    print(f"\n📝 Testing outputs against {GetWeatherInput.__name__} schema:")

    for i, output in enumerate(test_outputs, 1):
        try:
            validated = GetWeatherInput(**output)
            print(f"\n   {i}. ✅ Valid: {json.dumps(output)}")
            print(f"      → Validated: location='{validated.location}', unit='{validated.unit}'")
        except Exception as e:
            print(f"\n   {i}. ❌ Invalid: {json.dumps(output)}")
            print(f"      → Error: {str(e)}")

    # -------------------------------------------------------------------------
    # Example 5: Complete Usage Pattern
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Usage Pattern")
    print("=" * 80)

    print("""
# Step 1: Define your function schema as a Pydantic model
class MyFunctionInput(BaseModel):
    param1: str = Field(..., description="Description here")
    param2: int = Field(default=10, description="Another param")

# Step 2: Create SamplingParams with guided decoding
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    guided_decoding=GuidedDecodingParams.from_optional(json=MyFunctionInput)
)

# Step 3: Use with vLLM LLM
# from vllm import LLM
# llm = LLM(model="your-model")
# outputs = llm.generate(prompts=["Generate weather query"], sampling_params=sampling_params)

# Step 4: LLM output will automatically match your schema!
# Parse and validate: MyFunctionInput(**json.loads(output.outputs[0].text))
    """)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
✅ Function to Schema:
   - Possible but requires manual conversion using inspect module
   - Use function_to_json_schema() helper function

✅ Pydantic to Schema:
   - Built-in with Model.model_json_schema()
   - **RECOMMENDED APPROACH** for better type safety
   - Supports descriptions, defaults, enums, validation

✅ vLLM Integration:
   - Use GuidedDecodingParams.from_optional(json=YourModel)
   - Pass directly to SamplingParams(guided_decoding=...)
   - LLM will generate outputs matching your schema!

📚 Import Statements:
   from vllm.sampling_params import GuidedDecodingParams, SamplingParams
   from pydantic import BaseModel, Field
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
