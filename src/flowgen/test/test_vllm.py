#!/usr/bin/env python3
"""
Test vLLM basic functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flowgen import Agent
from flowgen.llm import vLLM
import asyncio


def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


async def test_vllm():
    """Test vLLM without streaming first."""
    print("=== Testing vLLM ===")
    
    # Initialize vLLM
    llm = vLLM(
        host="192.168.170.76",
        port="8077", 
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B"
    )
    
    print("1. Testing basic LLM call...")
    try:
        result = llm("what is 2+3?")
        print(f"LLM result: {result}")
    except Exception as e:
        print(f"LLM error: {str(e)}")
        return
    
    print("\n2. Testing Agent without streaming...")
    try:
        agent = Agent(
            llm=llm, 
            tools=[calculate], 
            stream=False,
            enable_rich_debug=True
        )
        
        result = await agent.run_async("what is 2+3?")
        print(f"Agent result: {result}")
    except Exception as e:
        print(f"Agent error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vllm())