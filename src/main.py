import ai
from mcp_tools.web import async_web_search
from mcp_tools.fetch import scrapling_get
from logger.logger import get_logger
log = get_logger(name="main", level="INFO", enable_rich=True, enable_files=True, subdir="main")
lm = ai.LM()
tools = [async_web_search, scrapling_get]
history = [{"role": "user", "content": input(">")}]
async def main():
    result = await ai.agent(lm=lm, history=history, tools=tools, max_iterations=5, logger=log)
import asyncio
asyncio.run(main())
