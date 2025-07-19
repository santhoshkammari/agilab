# Claude

> A powerful terminal-based chat application with integrated LLM capabilities and extensible tool support.

![img.png](img.png)

## Overview

Claude is a modern TUI (Terminal User Interface) chat application that provides seamless interaction with Large Language Models through Ollama. Built with Python and Textual, it offers a rich, responsive interface for AI-powered conversations with built-in tool integration for enhanced functionality.

## Key Features

- **Rich Terminal Interface**: Modern, responsive TUI built with Textual framework
- **LLM Integration**: Native support for Ollama with multiple model compatibility
- **Extensible Tools**: Built-in file operations and web utilities
- **Async Architecture**: Non-blocking operations for smooth user experience
- **Modular Design**: Clean, maintainable codebase with clear separation of concerns

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd claude

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

==========================================================================================
PERFORMANCE COMPARISON TABLE for 10 Queries
==========================================================================================
+------------------+------------+-----------+-----------+-----------+---------------+-------------------+
| Wait Until       | Headless   |   Avg (s) |   Min (s) |   Max (s) |   Std Dev (s) |   Success Queries |
+==================+============+===========+===========+===========+===============+===================+
| load             | True       |     0.894 |     0.669 |     1.212 |         0.155 |                10 |
+------------------+------------+-----------+-----------+-----------+---------------+-------------------+
| load             | False      |     0.897 |     0.672 |     1.099 |         0.13  |                10 |
+------------------+------------+-----------+-----------+-----------+---------------+-------------------+
| domcontentloaded | True       |     0.807 |     0.656 |     1.039 |         0.115 |                10 |
+------------------+------------+-----------+-----------+-----------+---------------+-------------------+
| domcontentloaded | False      |     0.996 |     0.724 |     2.656 |         0.589 |                10 |
+------------------+------------+-----------+-----------+-----------+---------------+-------------------+

🏆 Fastest: domcontentloaded, headless=True (0.807s avg)
🐌 Slowest: domcontentloaded, headless=False (0.996s avg)
⚡ Speed improvement: 0.189s (18.9% faster)


## Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: Local installation required
- **Dependencies**: textual, ollama, pydantic

## Architecture

```
claude/
├── core/           # Core application components
│   └── components/ # UI components and layouts
├── llm/           # Language model integrations
└── tools/         # Utility and tool implementations
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.
