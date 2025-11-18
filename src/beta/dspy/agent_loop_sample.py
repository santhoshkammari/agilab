"""
Agent loop demo showing the agent() function

The agent() function handles the multi-turn loop automatically:
1. Calls step() once per iteration
2. Waits for tool results
3. Adds to history
4. Repeats until no more tool calls or max_iterations reached
"""

import asyncio
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from agent import agent
from lm import LM


def get_weather(city: str, unit: str = "celsius"):
    """
    Get the current weather for a city

    Args:
        city: The name of the city
        unit: Temperature unit (choices: ["celsius", "fahrenheit"])
    """
    return f"Weather in {city}: 22 degrees {unit}"


def search(query: str) -> str:
    """
    Search for information

    Args:
        query: Search query
    """
    return f"Search results for '{query}': [mock results]"


async def demo_simple_loop():
    """Demo 1: Simple agent loop"""
    print("\n" + "="*60)
    print("DEMO 1: Simple Agent Loop")
    print("="*60)

    lm = LM()
    tools = [get_weather, search]

    result = await agent(
        lm=lm,
        initial_message="What is the weather in London and Paris? /no_think",
        tools=tools,
        max_iterations=5
    )

    print(f"Iterations: {result['iterations']}")
    print(f"Total tool calls: {result['tool_calls_total']}")
    print(f"\nFinal Response: {result['final_response'][:200]}...")

    print(f"\nConversation History ({len(result['history'])} messages):")
    for i, msg in enumerate(result['history']):
        role = msg.get("role", "unknown")
        if role == "assistant":
            content = msg.get("content", "")[:50] if msg.get("content") else "[no content]"
            print(f"  {i}: {role:10} → {content}...")
        elif role == "tool":
            content = msg.get("content", "")[:50]
            print(f"  {i}: {role:10} → {content}...")
        else:
            content = msg.get("content", "")[:50]
            print(f"  {i}: {role:10} → {content}...")


async def demo_multi_turn():
    """Demo 2: Multi-turn conversation"""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Turn Agent Loop")
    print("="*60)

    lm = LM()
    tools = [get_weather, search]

    result = await agent(
        lm=lm,
        initial_message="Search for Paris tourism info and check weather there /no_think",
        tools=tools,
        max_iterations=5,
        early_tool_execution=True
    )

    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {result['tool_calls_total']}")
    print(f"\nFinal response:\n{result['final_response']}")


async def demo_max_iterations():
    """Demo 3: Max iterations limit"""
    print("\n" + "="*60)
    print("DEMO 3: Max Iterations Limit")
    print("="*60)

    lm = LM()
    tools = [get_weather, search]

    result = await agent(
        lm=lm,
        initial_message="Get weather for London, Paris, Tokyo, and Sydney /no_think",
        tools=tools,
        max_iterations=2  # Limit to 2 iterations
    )

    print(f"Iterations: {result['iterations']} (max was 2)")
    print(f"Tool calls: {result['tool_calls_total']}")
    print(f"History length: {len(result['history'])} messages")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("AGENT LOOP FUNCTION DEMO")
    print("="*60)

    await demo_simple_loop()
    await demo_multi_turn()
    await demo_max_iterations()

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
