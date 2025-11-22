import ai
from mcp_tools.web import async_web_search
from logger.logger import get_logger

# Initialize logger and LM
log = get_logger(
    name="main",
    level="INFO",
    enable_rich=True,
    enable_files=True,
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


import asyncio
asyncio.run(main())

