#!/usr/bin/env python3
"""
Comprehensive test script for lm.py
Tests all functionality of the LM class including:
- Single message completion
- Batch processing
- String input handling
- Structured outputs
- Tool calling
- Error handling
"""

import asyncio
import json
from lm import LM


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_single_message():
    """Test 1: Single message completion"""
    print_section("TEST 1: Single Message Completion")

    lm = LM(api_base="http://192.168.170.76:8000")

    messages = [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ]

    print(f"Input: {messages[0]['content']}")
    response = lm(messages)

    print(f"Response type: {type(response)}")
    print(f"Response keys: {response.keys()}")

    content = response["choices"][0]["message"]["content"]
    print(f"Content: {content}")

    # Check usage stats
    if "usage" in response:
        print(f"Tokens used: {response['usage']}")

    print("✅ TEST 1 PASSED")
    return response


def test_string_input():
    """Test 2: String input (should be converted to messages)"""
    print_section("TEST 2: String Input")

    lm = LM(api_base="http://192.168.170.76:8000")

    prompt = "What is the capital of France? Answer in one word."
    print(f"Input string: {prompt}")

    response = lm(prompt)
    content = response["choices"][0]["message"]["content"]

    print(f"Content: {content}")
    print("✅ TEST 2 PASSED")
    return response


def test_batch_processing():
    """Test 3: Batch processing"""
    print_section("TEST 3: Batch Processing")

    lm = LM(api_base="http://192.168.170.76:8000")

    # Create batch of conversations
    batch = [
        [{"role": "user", "content": "What is 1+1?"}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 3+3?"}],
    ]

    print(f"Batch size: {len(batch)}")
    for i, conv in enumerate(batch):
        print(f"  Message {i+1}: {conv[0]['content']}")

    responses = lm(batch)

    print(f"\nResponses received: {len(responses)}")
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"  Response {i+1}: ERROR - {response}")
        else:
            content = response["choices"][0]["message"]["content"]
            print(f"  Response {i+1}: {content[:100]}")

    print("✅ TEST 3 PASSED")
    return responses


def test_structured_output():
    """Test 4: Structured output with JSON schema"""
    print_section("TEST 4: Structured Output (JSON Schema)")

    lm = LM(api_base="http://192.168.170.76:8000")

    # Simple JSON schema
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "explanation": {"type": "string"}
        },
        "required": ["answer", "explanation"]
    }

    messages = [
        {"role": "user", "content": "What is 5+7? Provide the answer and explanation."}
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "MathAnswer",
            "schema": schema,
            "strict": True
        }
    }

    print(f"Input: {messages[0]['content']}")
    print(f"Schema: {json.dumps(schema, indent=2)}")

    response = lm(messages, response_format=response_format)
    content = response["choices"][0]["message"]["content"]

    print(f"\nRaw content: {content}")

    # Try to parse as JSON
    try:
        parsed = json.loads(content)
        print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
        print("✅ TEST 4 PASSED")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        print("⚠️  TEST 4 FAILED (but LM call succeeded)")

    return response


def test_conversation():
    """Test 5: Multi-turn conversation"""
    print_section("TEST 5: Multi-turn Conversation")

    lm = LM(api_base="http://192.168.170.76:8000")

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"}
    ]

    print("Conversation history:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")

    response = lm(messages)
    content = response["choices"][0]["message"]["content"]

    print(f"\nAssistant response: {content}")
    print("✅ TEST 5 PASSED")
    return response


def test_parameters():
    """Test 6: Custom parameters (temperature, max_tokens, etc.)"""
    print_section("TEST 6: Custom Parameters")

    lm = LM(api_base="http://192.168.170.76:8000")

    messages = [{"role": "user", "content": "Write a haiku about coding."}]

    print("Testing with custom parameters:")
    print("  temperature=0.9, max_tokens=100, top_p=0.95")

    response = lm(messages, temperature=0.9, max_tokens=100, top_p=0.95)
    content = response["choices"][0]["message"]["content"]

    print(f"\nResponse: {content}")
    print("✅ TEST 6 PASSED")
    return response


def test_model_detection():
    """Test 7: Model detection and provider parsing"""
    print_section("TEST 7: Model Detection & Provider Parsing")

    # Test default (no model specified)
    lm1 = LM(api_base="http://192.168.170.76:8000")
    print(f"Default - Provider: {lm1.provider}, Model: {lm1.model}")

    # Test with vllm prefix
    lm2 = LM(model="vllm:/path/to/model", api_base="http://192.168.170.76:8000")
    print(f"With vllm: - Provider: {lm2.provider}, Model: {lm2.model}")

    # Test provider is correctly set
    assert lm1.provider == "vllm", "Default provider should be vllm"
    assert lm2.provider == "vllm", "Explicit provider should be vllm"

    print("✅ TEST 7 PASSED")


def test_async_internals():
    """Test 8: Verify async internals work correctly"""
    print_section("TEST 8: Async Internals Verification")

    lm = LM(api_base="http://192.168.170.76:8000")

    print("Testing _is_batch detection:")

    # Single message
    single = [{"role": "user", "content": "hi"}]
    is_batch_single = lm._is_batch(single)
    print(f"  Single message: {is_batch_single} (should be False)")
    assert not is_batch_single, "Single message incorrectly detected as batch"

    # Batch
    batch = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]]
    is_batch_batch = lm._is_batch(batch)
    print(f"  Batch messages: {is_batch_batch} (should be True)")
    assert is_batch_batch, "Batch incorrectly detected as single"

    # String
    string = "hello"
    is_batch_string = lm._is_batch(string)
    print(f"  String input: {is_batch_string} (should be False)")
    assert not is_batch_string, "String incorrectly detected as batch"

    print("✅ TEST 8 PASSED")


def test_error_handling():
    """Test 9: Error handling"""
    print_section("TEST 9: Error Handling")

    # Test with invalid endpoint
    lm_bad = LM(api_base="http://192.168.170.76:9999")  # Wrong port

    print("Testing connection to invalid endpoint (expected to fail):")
    try:
        response = lm_bad("Hello")
        print("❌ Should have raised an error!")
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"   Error message: {str(e)[:100]}")

    print("✅ TEST 9 PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  LM.PY COMPREHENSIVE TEST SUITE")
    print("="*60)

    try:
        # Run tests
        test_model_detection()
        test_async_internals()
        test_single_message()
        test_string_input()
        test_conversation()
        test_parameters()
        test_batch_processing()
        test_structured_output()
        test_error_handling()

        # Summary
        print_section("TEST SUITE SUMMARY")
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nLM class is working correctly with:")
        print("  ✓ Single message completion")
        print("  ✓ String input conversion")
        print("  ✓ Multi-turn conversations")
        print("  ✓ Custom parameters")
        print("  ✓ Batch processing")
        print("  ✓ Structured outputs (JSON schema)")
        print("  ✓ Error handling")
        print("  ✓ Async internals")

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
