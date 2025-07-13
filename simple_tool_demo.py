#!/usr/bin/env python3
"""
Simple demonstration of tool calling with Google GenAI.
Shows how to create and use a simple sum function.
"""

import sys
import os
import json

# Add claude directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude'))

def simple_sum(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    result = a + b
    print(f"🧮 simple_sum({a}, {b}) = {result}")
    return result

def simple_multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    result = a * b
    print(f"🧮 simple_multiply({a}, {b}) = {result}")
    return result

def test_tool_integration():
    """Demonstrate tool calling integration."""
    print("🔧 Tool Calling Demo with Google GenAI")
    print("=" * 50)
    
    try:
        from claude.llm import ChatGoogleGenAI
        from claude.llm.compat import LLMCompatibilityWrapper
        
        # Initialize Google GenAI
        raw_llm = ChatGoogleGenAI(
            model="gemini-2.5-flash",
            api_key="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE",
            thinking_budget=0
        )
        llm = LLMCompatibilityWrapper(raw_llm)
        
        # Test scenarios
        test_cases = [
            "What is 15 + 27?",
            "Calculate 8 * 6",
            "What's the sum of 100 and 200?",
            "Multiply 12 by 5"
        ]
        
        print("🤖 Testing various math questions...\n")
        
        for i, question in enumerate(test_cases, 1):
            print(f"Test {i}: {question}")
            
            # Send question to AI
            response = llm.chat([{"role": "user", "content": question}])
            print(f"AI: {response.message.content}")
            
            # Extract numbers and demonstrate our functions
            if "+" in question or "sum" in question.lower():
                # Try to extract numbers for sum
                numbers = [int(x) for x in question.split() if x.isdigit()]
                if len(numbers) >= 2:
                    actual = simple_sum(numbers[0], numbers[1])
                    print(f"Our function: {actual}")
            elif "*" in question or "multiply" in question.lower():
                # Try to extract numbers for multiplication
                numbers = [int(x) for x in question.split() if x.isdigit()]
                if len(numbers) >= 2:
                    actual = simple_multiply(numbers[0], numbers[1])
                    print(f"Our function: {actual}")
            
            print("-" * 30)
        
        print("\n✅ Tool calling demo completed!")
        print("💡 Next step: Integrate these functions directly into the chat interface")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tool_integration()