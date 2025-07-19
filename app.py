import asyncio
from pathlib import Path

from claude.core.input import ChatApp
import logging
logging.basicConfig(level=logging.WARNING)

async def main():
    """Initialize browser and start the app"""
    app = ChatApp(cwd=str(Path(__file__).parent))
    
    # Initialize browser before starting the app
    await app.browser.initialize()
    
    # Run the app using run_async
    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())