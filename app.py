from pathlib import Path

from claude.core.input import ChatApp

if __name__ == "__main__":
    app = ChatApp(cwd=str(Path(__file__).parent))
    app.run()