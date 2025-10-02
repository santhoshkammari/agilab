#!/usr/bin/env python3
"""
Pydantic Model to Grammar Conversion for vLLM
==============================================

This demonstrates the complete flow:
Pydantic Model → JSON Schema → Grammar → Token-level Validation
"""

import json
from pydantic import BaseModel, Field
from typing import Literal, Optional
import xgrammar as xgr
from vllm.sampling_params import GuidedDecodingParams


# ============================================================================
# EXAMPLE PYDANTIC MODELS
# ============================================================================

class GetWeatherInput(BaseModel):
    """Input schema for get_weather function."""
    location: str = Field(..., description="City name or location")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit"
    )


class Person(BaseModel):
    """Simple person schema."""
    name: str
    age: int
    city: str


class SearchQuery(BaseModel):
    """Search query with filters."""
    query: str = Field(..., description="Search query string")
    limit: int = Field(default=10, ge=1, le=100)
    filters: Optional[list[str]] = None


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def pydantic_to_json_schema(model: type[BaseModel]) -> dict:
    """
    Step 1: Convert Pydantic model to JSON schema.

    This is built into Pydantic via .model_json_schema()
    """
    return model.model_json_schema()


def json_schema_to_grammar(schema: dict) -> xgr.Grammar:
    """
    Step 2: Convert JSON schema to xgrammar Grammar object.

    This creates a grammar that can validate JSON structure.
    """
    schema_str = json.dumps(schema)
    return xgr.Grammar.from_json_schema(schema_str)


def compile_grammar_for_tokenizer(schema: dict, tokenizer) -> xgr.CompiledGrammar:
    """
    Step 3: Compile grammar for specific tokenizer.

    This creates optimized token validation rules.
    """
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    return compiler.compile_json_schema(json.dumps(schema))


def create_grammar_matcher(compiled_grammar: xgr.CompiledGrammar) -> xgr.GrammarMatcher:
    """
    Step 4: Create a matcher for token-by-token validation.

    This is used during LLM generation to validate each token.
    """
    return xgr.GrammarMatcher(compiled_grammar)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("PYDANTIC MODEL → GRAMMAR CONVERSION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Pydantic Model → JSON Schema
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Pydantic Model → JSON Schema")
    print("=" * 80)

    model = GetWeatherInput
    schema = pydantic_to_json_schema(model)

    print(f"\n📝 Pydantic Model: {model.__name__}")
    print(f"   Fields: {list(model.model_fields.keys())}")

    print(f"\n✅ JSON Schema:")
    print(json.dumps(schema, indent=2))

    # -------------------------------------------------------------------------
    # STEP 2: JSON Schema → Grammar Object
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: JSON Schema → Grammar Object")
    print("=" * 80)

    grammar = json_schema_to_grammar(schema)

    print(f"\n✅ Grammar created: {type(grammar)}")
    print(f"   Class: {grammar.__class__.__name__}")
    print(f"   Module: {grammar.__class__.__module__}")

    # -------------------------------------------------------------------------
    # STEP 3: Compile Grammar for Tokenizer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Compile Grammar for Tokenizer")
    print("=" * 80)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        compiled = compile_grammar_for_tokenizer(schema, tokenizer)

        print(f"\n✅ Grammar compiled for tokenizer: gpt2")
        print(f"   Compiled type: {type(compiled)}")
        print(f"   Class: {compiled.__class__.__name__}")

        # -------------------------------------------------------------------------
        # STEP 4: Create Grammar Matcher
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 4: Create Grammar Matcher (Token Validator)")
        print("=" * 80)

        matcher = create_grammar_matcher(compiled)

        print(f"\n✅ GrammarMatcher created: {type(matcher)}")
        print(f"   This validates tokens during LLM generation!")
        print(f"   Each token is checked: valid → accept, invalid → reject")

    except Exception as e:
        print(f"\n⚠️  Tokenizer step skipped: {e}")

    # -------------------------------------------------------------------------
    # WHAT THE GRAMMAR ENFORCES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("WHAT THE GRAMMAR ENFORCES")
    print("=" * 80)

    print(f"""
For the model: {model.__name__}

Grammar ensures output structure:
{{
  "location": "<string>",
  "unit": "celsius" | "fahrenheit"
}}

✅ VALID outputs (grammar accepts):
  {{"location": "Tokyo", "unit": "celsius"}}
  {{"location": "Paris", "unit": "fahrenheit"}}
  {{"location": "NYC"}}  ← uses default "celsius"

❌ INVALID outputs (grammar rejects):
  {{"city": "Tokyo", "unit": "celsius"}}  ← wrong key
  {{"location": "Tokyo", "unit": "kelvin"}}  ← invalid enum
  {{"location": 123, "unit": "celsius"}}  ← wrong type
  {{"location": "Tokyo", "temperature": "celsius"}}  ← extra key rejected
""")

    # -------------------------------------------------------------------------
    # VLLM INTEGRATION (THE EASY WAY)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VLLM INTEGRATION - THE EASY WAY")
    print("=" * 80)

    print("""
All the above steps are handled automatically by vLLM!

Just do this:
""")

    # This does ALL the above steps internally!
    guided_params = GuidedDecodingParams.from_optional(json=GetWeatherInput)

    print(f"""
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

# Pass Pydantic model directly - vLLM handles the rest!
sampling_params = SamplingParams(
    temperature=0.0,
    guided_decoding=GuidedDecodingParams.from_optional(json={model.__name__})
)

✅ Behind the scenes, vLLM does:
   1. {model.__name__}.model_json_schema() → JSON schema
   2. xgr.Grammar.from_json_schema() → Grammar
   3. Compile grammar for your model's tokenizer
   4. Create matcher for token validation
   5. Generate tokens that MUST match your Pydantic schema!
""")

    # -------------------------------------------------------------------------
    # OTHER GRAMMAR FORMATS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OTHER GRAMMAR FORMATS SUPPORTED")
    print("=" * 80)

    examples = [
        ("REGEX", r'\d{3}-\d{3}-\d{4}', xgr.Grammar.from_regex, "123-456-7890"),
        ("EBNF", 'root ::= "Hello " [A-Z][a-z]+ "!"', xgr.Grammar.from_ebnf, "Hello John!"),
        ("JSON Schema", '{"type": "string"}', lambda x: xgr.Grammar.from_json_schema(x), '"any text"'),
    ]

    for name, pattern, converter, example in examples:
        try:
            if name == "JSON Schema":
                grammar = converter(pattern)
            else:
                grammar = converter(pattern)
            print(f"\n✅ {name}:")
            print(f"   Pattern: {pattern}")
            print(f"   Example match: {example}")
            print(f"   Grammar type: {type(grammar).__name__}")
        except Exception as e:
            print(f"\n❌ {name}: {e}")

    # -------------------------------------------------------------------------
    # MULTIPLE MODEL EXAMPLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MULTIPLE MODEL EXAMPLES")
    print("=" * 80)

    models = [
        (GetWeatherInput, {"location": "Tokyo", "unit": "celsius"}),
        (Person, {"name": "Alice", "age": 30, "city": "NYC"}),
        (SearchQuery, {"query": "python", "limit": 5}),
    ]

    for model_class, example_output in models:
        schema = pydantic_to_json_schema(model_class)
        grammar = json_schema_to_grammar(schema)

        print(f"\n✅ {model_class.__name__}:")
        print(f"   Fields: {list(model_class.model_fields.keys())}")
        print(f"   Grammar: {type(grammar).__name__}")
        print(f"   Example output: {json.dumps(example_output)}")

        # Validate example
        validated = model_class(**example_output)
        print(f"   ✓ Validated successfully")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: COMPLETE CONVERSION PIPELINE")
    print("=" * 80)

    print("""
┌─────────────────┐
│ Pydantic Model  │  class GetWeatherInput(BaseModel): ...
└────────┬────────┘
         │ .model_json_schema()
         ▼
┌─────────────────┐
│  JSON Schema    │  {"type": "object", "properties": {...}}
└────────┬────────┘
         │ xgr.Grammar.from_json_schema()
         ▼
┌─────────────────┐
│  Grammar Object │  xgr.Grammar (abstract rules)
└────────┬────────┘
         │ xgr.GrammarCompiler.compile_json_schema()
         ▼
┌─────────────────┐
│ Compiled Grammar│  xgr.CompiledGrammar (tokenizer-specific)
└────────┬────────┘
         │ xgr.GrammarMatcher()
         ▼
┌─────────────────┐
│ Grammar Matcher │  Validates tokens during LLM generation
└─────────────────┘
         │
         ▼
    ✅ STRUCTURED OUTPUT!

Key takeaways:
• Pydantic models → JSON schema (built-in)
• JSON schema → Grammar (xgrammar)
• Grammar → Token validation (enforces structure)
• vLLM handles everything automatically!

Use: GuidedDecodingParams.from_optional(json=YourModel)
""")


if __name__ == "__main__":
    main()
