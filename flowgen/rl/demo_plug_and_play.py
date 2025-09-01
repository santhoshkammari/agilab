#!/usr/bin/env python3
"""
Demonstration of plug-and-play RL environments with any LLM.

This script shows how to:
1. Create domain-specific environments (MathsEnv, WebSearchEnv, PythonExecutionEnv)
2. Attach any LLM to train on that environment 
3. Use the trained model with the same tools during inference

The key insight: Models learn tool usage patterns during RL training,
then transfer that knowledge to inference when given the same tools.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from environments import MathsEnv, WebSearchEnv, PythonExecutionEnv, create_environment
from reward import reward_exact_match, reward_step_by_step


def create_sample_datasets():
    """Create sample datasets for each domain."""
    
    # Math dataset
    math_data = Dataset.from_list([
        {"prompt": "What is 15 * 23?", "answer": "345"},
        {"prompt": "Calculate the factorial of 6", "answer": "720"},
        {"prompt": "Is 17 a prime number?", "answer": "Yes, 17 is prime"},
        {"prompt": "Solve for x: 2x + 5 = 13", "answer": "x = 4"},
        {"prompt": "What is 8^2 + 3*4?", "answer": "76"}
    ])
    
    # Web search dataset  
    web_data = Dataset.from_list([
        {"prompt": "Find information about machine learning trends in 2024", "answer": "ML trends include LLMs, multimodal AI, and autonomous agents"},
        {"prompt": "Search for Python programming best practices", "answer": "Best practices include PEP 8, type hints, and unit testing"},
        {"prompt": "What are the latest developments in quantum computing?", "answer": "Recent advances in quantum error correction and practical applications"}
    ])
    
    # Python coding dataset
    python_data = Dataset.from_list([
        {"prompt": "Write a function to calculate fibonacci numbers", "answer": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"},
        {"prompt": "Create a function to reverse a string", "answer": "def reverse_string(s): return s[::-1]"},
        {"prompt": "Write code to find the maximum element in a list", "answer": "def find_max(lst): return max(lst) if lst else None"}
    ])
    
    return math_data, web_data, python_data


def demo_environment_creation():
    """Demonstrate creating different environments."""
    print("=== Creating Domain-Specific Environments ===\n")
    
    math_data, web_data, python_data = create_sample_datasets()
    
    # Method 1: Direct instantiation
    print("1. Direct Environment Creation:")
    math_env = MathsEnv(dataset=math_data, difficulty_level="basic")
    web_env = WebSearchEnv(dataset=web_data, max_search_results=3)
    python_env = PythonExecutionEnv(dataset=python_data, timeout=10)
    
    print(f"✓ MathsEnv: {len(math_env.tools)} tools, {len(math_env.reward_funcs)} rewards")
    print(f"✓ WebSearchEnv: {len(web_env.tools)} tools, {len(web_env.reward_funcs)} rewards") 
    print(f"✓ PythonExecutionEnv: {len(python_env.tools)} tools, {len(python_env.reward_funcs)} rewards\n")
    
    # Method 2: Factory pattern
    print("2. Factory Pattern Creation:")
    math_env2 = create_environment('math', math_data, difficulty_level='advanced')
    web_env2 = create_environment('web', web_data, max_search_results=5)
    python_env2 = create_environment('python', python_data, timeout=15)
    
    print(f"✓ Factory MathsEnv: {type(math_env2).__name__}")
    print(f"✓ Factory WebSearchEnv: {type(web_env2).__name__}")
    print(f"✓ Factory PythonExecutionEnv: {type(python_env2).__name__}\n")
    
    return math_env, web_env, python_env


def demo_tool_execution():
    """Demonstrate tool execution in each environment."""
    print("=== Testing Environment Tools ===\n")
    
    math_env, web_env, python_env = demo_environment_creation()
    
    # Test MathsEnv tools
    print("1. MathsEnv Tool Testing:")
    print(f"Calculator: {math_env.calculator('25 * 4 + 10')}")
    print(f"Prime check: {math_env.check_prime('29')}")
    print(f"Factorial: {math_env.factorial('7')}\n")
    
    # Test WebSearchEnv tools
    print("2. WebSearchEnv Tool Testing:")
    search_result = web_env.web_search("artificial intelligence", 2)
    print(f"Search results: {search_result[:100]}...")
    
    extract_result = web_env.extract_info("Machine learning and AI are transforming industries", "AI")
    print(f"Information extraction: {extract_result}\n")
    
    # Test PythonExecutionEnv tools
    print("3. PythonExecutionEnv Tool Testing:")
    code_result = python_env.execute_code("for i in range(3): print(f'Hello {i}')")
    print(f"Code execution result:\n{code_result}")
    
    analysis_result = python_env.analyze_code("def greet(name):\n    print(f'Hello {name}')")
    print(f"Code analysis: {analysis_result}\n")


def demo_llm_environment_interaction():
    """Demonstrate LLM interacting with environments."""
    print("=== LLM + Environment Interaction ===\n")
    
    # Create a simple mock LLM for demonstration
    class MockLLM:
        def __init__(self, name):
            self.name = name
            
        def __call__(self, messages, tools=None, **kwargs):
            """Mock LLM that demonstrates tool usage."""
            last_message = messages[-1]['content'] if messages else ""
            
            # Mock responses that would trigger tool usage
            if "calculator" in last_message.lower() or "calculate" in last_message.lower():
                return {
                    'content': 'I need to use the calculator tool.',
                    'tool_calls': [{
                        'id': 'call_1',
                        'type': 'function',
                        'function': {
                            'name': 'calculator',
                            'arguments': '{"expression": "15 * 23"}'
                        }
                    }]
                }
            elif "prime" in last_message.lower():
                return {
                    'content': 'Let me check if this number is prime.',
                    'tool_calls': [{
                        'id': 'call_2', 
                        'type': 'function',
                        'function': {
                            'name': 'check_prime',
                            'arguments': '{"number": "17"}'
                        }
                    }]
                }
            else:
                return {'content': f'This is {self.name} responding to: {last_message}'}
    
    # Test environments with mock LLM
    math_data, _, _ = create_sample_datasets()
    
    print("1. MathsEnv with Mock LLM:")
    math_env = MathsEnv(dataset=math_data.select(range(2)))  # Just 2 examples
    mock_llm = MockLLM("MockGPT")
    
    # Single rollout
    result = math_env.rollout(mock_llm, "Calculate 15 * 23")
    print(f"Rollout result: {result['completion']}")
    print(f"Reward: {result['reward']}")
    print(f"Tool calls made: {len([msg for msg in result['messages'] if msg['role'] == 'tool'])}\n")
    
    # Evaluate on dataset
    print("2. Dataset Evaluation:")
    eval_results = math_env.evaluate(mock_llm, num_examples=2)
    print(f"Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"Examples evaluated: {eval_results['num_examples']}\n")


def demo_training_pipeline():
    """Demonstrate the training pipeline concept."""
    print("=== Training Pipeline Demonstration ===\n")
    
    # Mock trainer for demonstration
    class MockTrainer:
        def __init__(self, model_name):
            self.model_name = model_name
            self.train_dataset = None
            self.reward_funcs = None
            self.configured = False
            
        def train(self):
            if not self.configured:
                return "Error: Trainer not configured with environment"
            dataset_len = len(self.train_dataset) if self.train_dataset else 0
            reward_len = len(self.reward_funcs) if self.reward_funcs else 0
            return f"✓ {self.model_name} trained on {dataset_len} examples with {reward_len} reward functions"
    
    math_data, web_data, python_data = create_sample_datasets()
    
    print("1. Training Pipeline Setup:")
    
    # Create environments
    math_env = MathsEnv(dataset=math_data)
    web_env = WebSearchEnv(dataset=web_data) 
    python_env = PythonExecutionEnv(dataset=python_data)
    
    # Create trainers
    math_trainer = MockTrainer("Qwen-Math")
    web_trainer = MockTrainer("Llama-Web")
    python_trainer = MockTrainer("CodeLlama")
    
    print("✓ Environments and trainers created\n")
    
    print("2. Environment >> Trainer Piping:")
    
    # Pipe environments to trainers
    configured_math_trainer = math_env >> math_trainer
    configured_web_trainer = web_env >> web_trainer  
    configured_python_trainer = python_env >> python_trainer
    
    configured_math_trainer.configured = True
    configured_web_trainer.configured = True
    configured_python_trainer.configured = True
    
    print("✓ Environments piped to trainers\n")
    
    print("3. Training Results:")
    print(configured_math_trainer.train())
    print(configured_web_trainer.train())
    print(configured_python_trainer.train())
    print()


def demo_inference_with_tools():
    """Demonstrate inference using trained models with tools."""
    print("=== Inference with Environment Tools ===\n")
    
    # Simulate trained models that know how to use tools
    class TrainedModel:
        def __init__(self, name, specialized_domain):
            self.name = name
            self.domain = specialized_domain
            
        def __call__(self, prompt, tools=None, **kwargs):
            """Simulate a model trained on specific environment tools."""
            if tools and self.domain == "math":
                # Model trained on MathsEnv knows to use calculator
                if any("calculator" in str(tool) for tool in tools):
                    return f"[{self.name}] I'll use the calculator: 15 * 23 = 345"
            elif tools and self.domain == "web":
                # Model trained on WebSearchEnv knows to search
                return f"[{self.name}] Searching the web for relevant information..."
            elif tools and self.domain == "python":
                # Model trained on PythonExecutionEnv knows to execute code
                return f"[{self.name}] Let me run this Python code for you..."
            
            return f"[{self.name}] Standard response without tools"
    
    # Create environments to get their tools
    math_env = MathsEnv()
    web_env = WebSearchEnv()
    python_env = PythonExecutionEnv()
    
    # Simulate trained models
    math_model = TrainedModel("Qwen-Math-Aligned", "math")
    web_model = TrainedModel("Llama-Web-Aligned", "web")  
    python_model = TrainedModel("CodeLlama-Python-Aligned", "python")
    
    print("1. Models trained on different environments:")
    print(f"✓ {math_model.name} (trained on MathsEnv)")
    print(f"✓ {web_model.name} (trained on WebSearchEnv)")
    print(f"✓ {python_model.name} (trained on PythonExecutionEnv)\n")
    
    print("2. Inference with environment tools:")
    
    # Math model with math tools
    math_result = math_model("Calculate 15 * 23", tools=math_env.tools)
    print(f"Math query: {math_result}")
    
    # Web model with web tools  
    web_result = web_model("Find info about AI", tools=web_env.tools)
    print(f"Web query: {web_result}")
    
    # Python model with python tools
    python_result = python_model("Write a function", tools=python_env.tools)
    print(f"Python query: {python_result}\n")
    
    print("3. Cross-domain usage (suboptimal but works):")
    # What happens when you use wrong tools?
    cross_result = math_model("Search for information", tools=web_env.tools)
    print(f"Math model with web tools: {cross_result}\n")


def main():
    """Run all demonstrations."""
    print("🚀 Plug-and-Play RL Environment Demo\n")
    print("=" * 50)
    
    try:
        demo_environment_creation()
        demo_tool_execution()
        demo_llm_environment_interaction()
        demo_training_pipeline()
        demo_inference_with_tools()
        
        print("=" * 50)
        print("🎉 Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("1. Each environment has specialized tools for its domain")
        print("2. Models learn tool usage patterns during RL training")
        print("3. Trained models transfer tool knowledge to inference")
        print("4. Plug-and-play: any LLM + any environment + any trainer")
        print("5. Tool specialization improves performance in specific domains")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()