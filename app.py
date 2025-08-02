import asyncio
from pathlib import Path

from claude.core.input import ChatApp
import logging
logging.basicConfig(level=logging.ERROR)

async def main():
    """Start the app with lazy initialization"""
    app = ChatApp(cwd=str(Path(__file__).parent))
    
    # Run the app using run_async (browser will be initialized on first use)
    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())