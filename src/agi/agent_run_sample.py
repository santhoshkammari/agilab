"""
Agent Framework Usage Examples

Demonstrates the agent.py framework for streaming-based agent execution
with async tool calling and early execution optimization.
"""

import asyncio
from agent import step, gen, tool_result_to_message, AssistantResponse, ToolCall
from lm import LM


def get_weather(city: str, unit: str = "celsius"):
    """
    Get the current weather for a city

    Args:
        city: The name of the city
        unit: Temperature unit (choices: ["celsius", "fahrenheit"])
    """
    return f"Weather in {city}: 22 degrees {unit}"


async def example_multi_turn_agent_loop():
    """
    Example 1: Multi-turn agent loop with early tool execution (default)

    Demonstrates:
    - Single step() calls for LLM generation
    - Async tool execution and waiting for results
    - Multi-turn conversation handling
    - Early tool execution optimization (enabled by default)
    """
    lm = LM()
    history = [
        {"role": "user", "content": "what is weather in london and canada?, do two parallel tool calls to get the weather in both cities /no_think"}
    ]
    tools = [get_weather]

    print("=== Multi-turn Agent Loop Demo (Early Tool Execution Enabled) ===\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Single LLM generation with async tool execution
        result = await step(lm=lm, history=history, tools=tools, early_tool_execution=True)

        print(f"Assistant message: {result.message}")
        print(f"Tool calls: {len(result.tool_calls)}")

        # Wait for tool execution (tools running in parallel)
        tool_results = await result.tool_results()

        # Add to history
        history.append(result.message)
        for tr in tool_results:
            print(f"Tool result: {tr}")
            history.append(tool_result_to_message(tr))

        # Stop if no more tool calls
        if not result.tool_calls:
            print(f"\nFinal response: {result.message.get('content', '')}")
            break

        if iteration > 10:
            print("\nMax iterations reached!")
            break

    print("\n=== Conversation History ===")
    for i, msg in enumerate(history):
        print(f"{i}: {msg}")


async def example_streaming_gen():
    """
    Example 3: Low-level streaming generation

    Demonstrates:
    - Direct use of gen() function for streaming LLM responses
    - Receiving raw AssistantResponse and ToolCall chunks
    - Manual handling of streaming chunks
    """
    lm = LM()
    messages = [{"role": "user", "content": "what is weather in london and canada?, do two parallel tool calls to get the weather in both cities /no_think"}]
    tools = [get_weather]

    print("\n\n=== Low-level Streaming Demo ===\n")

    async for chunk in gen(lm=lm, history=messages, tools=tools):
        if isinstance(chunk, AssistantResponse):
            print(f"[Text] {chunk.content}", end="", flush=True)
        elif isinstance(chunk, ToolCall):
            print(f"\n[Tool] {chunk.name} called with ID {chunk.id}")
            print(f"       Args: {chunk.arguments}")


async def main():
    """Run all examples"""
    # Example 1: Multi-turn with early execution (recommended)
    await example_multi_turn_agent_loop()

    # Example 3: Low-level streaming
    await example_streaming_gen()


if __name__ == "__main__":
    asyncio.run(main())
