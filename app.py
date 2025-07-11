from pathlib import Path

from claude.core.input import ChatApp
import logging
logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    app = ChatApp(cwd=str(Path(__file__).parent))
    app.run()