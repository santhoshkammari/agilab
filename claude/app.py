import asyncio
from pathlib import Path

from input_efficient import ChatApp
from logger import get_logger

logger = get_logger(__name__)

async def main():
    """Start the app with lazy initialization"""
    logger.debug("Starting Claude Code application")
    app = ChatApp(cwd=str(Path(__file__).parent))
    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())
