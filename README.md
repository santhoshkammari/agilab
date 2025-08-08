# Claude

> A powerful terminal-based chat application with integrated LLM capabilities and extensible tool support.

## Overview

Claude is a sophisticated terminal-based chat application that brings Claude Code's capabilities to your local environment. Built with Python and Textual, it provides a rich TUI interface for seamless AI conversations with comprehensive tool integration and multi-provider LLM support.

## Key Features

- **Modern TUI Interface**: Rich terminal interface with Gruvbox theming and markdown rendering
- **Multi-Provider Support**: Works with Gemini, Ollama, OpenRouter, and vLLM backends
- **Extensive Tool Integration**: File operations, web search, markdown analysis, and bash execution
- **Interactive Permissions**: Smart tool execution with configurable permission modes
- **File Autocomplete**: Tab-completion for file paths with fuzzy matching
- **Command History**: Persistent history with navigation shortcuts
- **Async Architecture**: Non-blocking operations with real-time status updates

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd claude

# Install dependencies
pip install -r requirements.txt

# Run the application
python claude/app.py
```

## Usage

The application starts with a welcome screen and supports various interaction modes:

- **Default Mode**: Standard operation with permission prompts for file operations
- **Auto-Accept Edits**: Automatically approve file editing operations
- **Bypass Permissions**: Skip all permission prompts
- **Plan Mode**: Planning mode for task organization

### Keyboard Shortcuts

- `Shift+Tab`: Cycle through permission modes
- `Tab`: Trigger file autocomplete
- `Ctrl+C`: Clear current input
- `Escape`: Interrupt conversation or hide autocomplete
- `Ctrl+L`: Clear screen
- `Shift+Up/Down`: Navigate command history

### Slash Commands

- `/provider`: Switch between LLM providers
- `/clear`: Clear conversation history
- `/help`: Show help information
- `/status`: Display current configuration

## Architecture

The application consists of two main components:

### `claude/app.py`
Simple application entry point that initializes the chat interface with the current working directory.

### `claude/chat.py`
Core application logic featuring:
- **ChatApp Class**: Main TUI application with Textual framework
- **Tool Integration**: Comprehensive tool ecosystem including:
  - File operations (read, write, edit, multi-edit)
  - Web utilities (search, fetch)
  - Markdown analysis
  - Bash command execution
  - Directory operations
- **Permission System**: Configurable tool execution permissions
- **Provider Management**: Support for multiple LLM backends
- **Rich UI Components**: Status indicators, autocomplete, and styled output

## Prerequisites

- **Python**: 3.8 or higher
- **LLM Provider**: At least one of Gemini, Ollama, OpenRouter, or vLLM configured
- **Dependencies**: textual, rich, asyncio, and provider-specific packages

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.
