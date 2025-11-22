import ai
from mcp_tools.web import async_web_search

# Now you can add async_web_search as a tool!
lm = ai.LM()
tools = [async_web_search]
history = [{"role": "user", "content": "What is the weather in London and Paris? Also search the web for latest news about AI. /no_think"}]
history = [{"role": "user", "content": "What is the weather in London /no_think"}]

async def main():
    result = await ai.agent(
    lm=lm,
    history=history,
    tools=tools,
    max_iterations=5
)
    print(f"Iterations: {result['iterations']}")
    print(f"Total tool calls: {result['tool_calls_total']}")
    print(f"\nFinal Response: {result['final_response']}...")


import asyncio
asyncio.run(main())

