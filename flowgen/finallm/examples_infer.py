from enum import Enum

from ollama_main import Ollama
from vllm_main import vLLM
from gemini_main import Gemini
from huggingface_main import hfLLM
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random


# Test schemas
class Person(BaseModel):
    name: str
    age: int
    is_available: bool

class MenuItem(BaseModel):
    course_name: str
    is_vegetarian: bool

class Restaurant(BaseModel):
    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]

# Test tool functions
def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    temperatures = [15, 18, 22, 25, 28, 30, 33]
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    return {
        "city": city,
        "temperature": random.choice(temperatures),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80)
    }

def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return int(a) + int(b)

def get_current_time(timezone: str) -> dict:
    """Get current time for timezone"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone
    }


# Generic test functions - works with any LLM
def test_basic_chat(llm):
    """Test basic chat functionality"""
    print("=== Basic Chat Test ===")
    try:
        response = llm("Tell me a short joke about programming")
        print(f"Content: {response['content']}")
        print(f"Think: {response.get('think', 'N/A')}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_streaming(llm):
    """Test streaming functionality"""
    print("\n=== Streaming Test ===")
    try:
        response = llm("Tell me a short story about AI", stream=True)
        if hasattr(response, '__iter__') and not isinstance(response, dict):
            # It's a generator/iterator
            print("Streaming output:")
            for chunk in response:
                print(chunk.get('content', ''), end='', flush=True)
            print("\n")
        else:
            # It's a dict (non-streaming response)
            print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_tool_calling(llm):
    """Test tool calling functionality"""
    print("\n=== Tool Calling Test ===")
    try:
        response = llm("What's the weather like in Tokyo?", tools=[get_weather])
        print(f"Content: {response['content']}")
        print(f"Tool calls: {response.get('tool_calls', [])}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_streaming_with_tools(llm):
    """Test streaming with tool calling"""
    print("\n=== Streaming + Tools Test ===")
    try:
        response = llm("What's 25 plus 17?", tools=[add_numbers], stream=True)
        if hasattr(response, '__iter__') and not isinstance(response, dict):
            # Streaming
            full_content = ""
            tool_calls = []
            for chunk in response:
                content = chunk.get('content', '')
                full_content += content
                print(content, end='', flush=True)
                if chunk.get('tool_calls'):
                    tool_calls.extend(chunk['tool_calls'])
            print(f"\nTool calls: {tool_calls}")
        else:
            # Non-streaming
            print(f"Content: {response['content']}")
            print(f"Tool calls: {response.get('tool_calls', [])}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_structured_output(llm):
    """Test structured output generation"""
    print("\n=== Structured Output Test ===")
    try:
        response = llm("Generate info about Alice, age 25, who is available", format=Person)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_complex_structured(llm):
    """Test complex structured output"""
    print("\n=== Complex Structured Test ===")
    try:
        response = llm("Generate a Mexican restaurant in Miami with 2 menu items", format=Restaurant)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_message_history(llm):
    """Test conversation with message history"""
    print("\n=== Message History Test ===")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        response = llm(messages)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_all_tests(llm, llm_name="LLM"):
    """Run all tests for any LLM"""
    print(f"\n{'='*50}")
    print(f"Testing {llm_name}")
    print(f"{'='*50}")
    
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Streaming", test_streaming),
        ("Tool Calling", test_tool_calling),
        ("Streaming + Tools", test_streaming_with_tools),
        ("Structured Output", test_structured_output),
        ("Complex Structured", test_complex_structured),
        ("Message History", test_message_history),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(llm)
        except Exception as e:
            print(f"Test {test_name} failed with: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*30}")
    print(f"Test Results for {llm_name}:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    return results


def test_ollama():
    """Test Ollama with specific configuration"""
    llm = Ollama(model='qwen3:1.7b', host="192.168.170.76:11434")
    return run_all_tests(llm, 'Ollama')

def test_vllm():
    """Test vLLM with specific configuration"""
    llm = vLLM(
        base_url="http://192.168.170.76:8077/v1",
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B"
    )
    return run_all_tests(llm, 'vLLM')

def test_gemini():
    """Test Gemini with specific configuration"""
    llm = Gemini(model='gemini-2.5-flash-lite')
    return run_all_tests(llm, 'Gemini')

def test_huggingface():
    """Test hfLLM with specific configuration"""
    llm = hfLLM(model='HuggingFaceTB/SmolLM2-135M-Instruct')
    return run_all_tests(llm, 'hfLLM')


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def vllm_structure_output():
    llm = vLLM(
        base_url="http://192.168.170.76:8077/v1",
        model="/home/ng6309/datascience/santhosh/models/Qwen__Qwen3-4B-Thinking-2507"
    )
    response = llm("Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
                   format=CarDescription)
    print(response)




if __name__ == "__main__":
    vllm_structure_output()
    exit()
    print("Running all LLM provider tests...\n")
    
    all_results = {}
    
    # Test Ollama
    print("Starting Ollama tests...")
    try:
        all_results['Ollama'] = test_ollama()
    except Exception as e:
        print(f"Ollama tests failed: {e}")
        all_results['Ollama'] = None
    
    # Test vLLM
    print("\nStarting vLLM tests...")
    try:
        all_results['vLLM'] = test_vllm()
    except Exception as e:
        print(f"vLLM tests failed: {e}")
        all_results['vLLM'] = None
    
    # Test Gemini
    print("\nStarting Gemini tests...")
    try:
        all_results['Gemini'] = test_gemini()
    except Exception as e:
        print(f"Gemini tests failed: {e}")
        all_results['Gemini'] = None
    
    # Test hfLLM
    print("\nStarting hfLLM tests...")
    try:
        all_results['hfLLM'] = test_huggingface()
    except Exception as e:
        print(f"hfLLM tests failed: {e}")
        all_results['hfLLM'] = None
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - ALL PROVIDERS")
    print(f"{'='*60}")
    
    for provider, results in all_results.items():
        print(f"\n{provider}:")
        if results is None:
            print("  ✗ PROVIDER FAILED TO INITIALIZE")
        else:
            for test_name, passed in results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {test_name}: {status}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
