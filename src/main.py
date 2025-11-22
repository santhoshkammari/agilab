import ai
from mcp_tools.web import async_web_search
from logger.logger import get_logger

# Initialize logger and LM
log = get_logger(
    name="main",
    level="DEBUG",
    enable_rich=True,
    enable_files=False,
    subdir="main"
)
lm = ai.LM()
tools = [async_web_search]
history = [{"role": "user", "content": "What is the weather in London /no_think"}]

async def main():
    result = await ai.agent(
    lm=lm,
    history=history,
    tools=tools,
    max_iterations=5,
    logger=log
)
    print(f"Iterations: {result['iterations']}")
    print(f"Total tool calls: {result['tool_calls_total']}")
    print(f"\nFinal Response: {result['final_response']}...")


import asyncio
asyncio.run(main())

