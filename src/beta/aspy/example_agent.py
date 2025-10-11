"""
Example: Using the new Agent class
"""
import asyncio
import aspy as a

# Configure LM
lm = a.LM(api_base="http://192.168.170.76:8000")
a.configure(lm=lm)


async def example_simple_call():
    """Simple call - still streams internally"""
    print("=== Simple Call ===")

    agent = a.Agent(system_prompt="You are a helpful math tutor.")

    # Simple call
    result = await agent("What is 15 + 27?")
    print(f"Result: {result}")
    print()


async def example_streaming():
    """Explicit streaming to see events"""
    print("=== Streaming Events ===")

    agent = a.Agent(system_prompt="You are a helpful coding assistant.")

    # Stream events
    async for event in agent.stream("Explain what is async/await in Python in one sentence"):
        print(f"[{event.type}]", end=" ")

        if event.type == "content":
            print(event.content)
        elif event.type == "message_start":
            print("Agent starting...")
        elif event.type == "message_end":
            print(f"Done! Model: {event.metadata.get('model')}")
        elif event.type == "error":
            print(f"Error: {event.content}")
    print()


async def example_conversation():
    """Multi-turn conversation"""
    print("=== Conversation ===")

    agent = a.Agent(system_prompt="You are a friendly assistant.")

    # Turn 1
    response1 = await agent("My name is Alice")
    print(f"Agent: {response1}")

    # Turn 2 - agent remembers
    response2 = await agent("What's my name?")
    print(f"Agent: {response2}")
    print()


async def example_custom_lm():
    """Agent with custom LM (not using global config)"""
    print("=== Custom LM ===")

    custom_lm = a.LM(api_base="http://192.168.170.76:8000")
    agent = a.Agent(
        system_prompt="You are a concise assistant.",
        lm=custom_lm
    )

    result = await agent("Say hello in one word")
    print(f"Result: {result}")
    print()


async def main():
    await example_simple_call()
    await example_streaming()
    await example_conversation()
    await example_custom_lm()


if __name__ == "__main__":
    asyncio.run(main())
