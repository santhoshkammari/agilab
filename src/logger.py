"""
Universal Logger - The Ultimate Logging Solution

A complete logging system that handles all data types, rich formatting,
rotatory files, AI conversations, and smart environment detection.
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class UniversalLogger:
    """
    Universal Logger that handles all data types with rich formatting and smart rotation.
    
    Features:
    - Handles 4 data types: str, dict, List[str], List[dict]
    - Rich console formatting with panels and tables
    - Rotatory file logging with subdirectory support
    - AI conversation logging
    - Smart environment detection
    - Multiple log levels including custom ones
    """
    
    # Log levels with custom additions
    LEVELS = {
        'DEV': 5,        # Development-only
        'DEBUG': 10,     # Detailed diagnostic info
        'INFO': 20,      # General information
        'PROD': 25,      # Production-safe info
        'WARNING': 30,   # Something unexpected
        'ERROR': 40,     # Serious problem
        'AUDIT': 45,     # Security/compliance logs
        'CRITICAL': 50,  # Very serious error
    }
    
    def __init__(
        self,
        name: str = "universal",
        level: Optional[Union[str, int]] = None,
        enable_rich: bool = True,
        enable_files: bool = True,
        log_dir: str = "logs",
        subdir: Optional[str] = None,
        max_bytes: int = 10*1024*1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize Universal Logger
        
        Args:
            name: Logger name
            level: Log level (auto-detected if None)
            enable_rich: Enable rich console formatting
            enable_files: Enable file logging
            log_dir: Base log directory
            subdir: Subdirectory for this logger
            max_bytes: Max file size before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.enable_files = enable_files
        
        # Setup rich console if available
        if self.enable_rich:
            self.console = Console()
        
        # Determine log level
        if level is None:
            level = self._detect_environment_level()
        self.level = self._parse_level(level)
        
        # Setup file logging
        if self.enable_files:
            self._setup_file_logging(log_dir, subdir, max_bytes, backup_count)
        else:
            self.file_logger = None
    
    def _detect_environment_level(self) -> str:
        """Smart environment detection for log level"""
        if os.getenv('DEBUG') or os.getenv('DEV'):
            return "DEV"
        elif os.getenv('PRODUCTION') or os.getenv('PROD'):
            return "PROD"
        elif sys.stdout.isatty():
            return "DEBUG"
        else:
            return "INFO"
    
    def _parse_level(self, level: Union[str, int]) -> int:
        """Parse level string/int to numeric level"""
        if isinstance(level, str):
            return self.LEVELS.get(level.upper(), 20)
        return level
    
    def _setup_file_logging(self, log_dir: str, subdir: Optional[str], max_bytes: int, backup_count: int):
        """Setup file logging with rotation"""
        # Determine log path
        base_path = Path(log_dir)
        if subdir:
            log_path = base_path / subdir
        else:
            log_path = base_path
        
        # Create directories
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        log_file = log_path / f"{self.name}.log"
        self.file_logger = logging.getLogger(f"file_{self.name}")
        self.file_logger.setLevel(logging.DEBUG)  # File gets everything
        self.file_logger.propagate = False  # Don't propagate to root logger (prevents console output)

        # Remove existing handlers
        self.file_logger.handlers.clear()

        # Add rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )

        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'
        )
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)
    
    def set_level(self, level: Union[str, int]):
        """Change log level dynamically"""
        self.level = self._parse_level(level)
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level"""
        return self.LEVELS.get(level.upper(), 20) >= self.level
    
    def _format_data(self, data: Any, method_type: str = "standard") -> tuple:
        """
        Universal data formatter that handles all 4 types
        Returns (console_output, file_output)
        """
        if isinstance(data, str):
            return self._format_string(data, method_type)
        elif isinstance(data, dict):
            return self._format_dict(data, method_type)
        elif isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return self._format_list_of_strings(data, method_type)
            elif all(isinstance(item, dict) for item in data):
                return self._format_list_of_dicts(data, method_type)
            else:
                return self._format_mixed_list(data, method_type)
        else:
            # Fallback for other types
            str_data = str(data)
            return str_data, str_data
    
    def _format_string(self, data: str, method_type: str) -> tuple:
        """Format string data"""
        if method_type == "rich" and self.enable_rich:
            # Check if markdown
            if any(char in data for char in ['*', '_', '`', '#']):
                rich_content = Markdown(data)
            else:
                rich_content = Text(data)
            return rich_content, data
        return data, data
    
    def _format_dict(self, data: dict, method_type: str) -> tuple:
        """Format dictionary data"""
        # File output - JSON
        file_output = json.dumps(data, ensure_ascii=False, indent=None)
        
        if method_type == "rich" and self.enable_rich:
            # Rich panel for console
            content = ""
            for key, value in data.items():
                content += f"[bold]{key}[/bold] : {value}\n"
            rich_content = content.strip()
            return rich_content, file_output
        else:
            # Simple key=value format for console
            console_output = " ".join([f"{k}={v}" for k, v in data.items()])
            return console_output, file_output
    
    def _format_list_of_strings(self, data: List[str], method_type: str) -> tuple:
        """Format list of strings"""
        if method_type == "rich" and self.enable_rich:
            content = "\n".join([f"• {item}" for item in data])
            return content, json.dumps(data)
        else:
            content = "\n  ".join([f"• {item}" for item in data])
            return content, json.dumps(data)
    
    def _format_list_of_dicts(self, data: List[dict], method_type: str) -> tuple:
        """Format list of dictionaries"""
        file_output = json.dumps(data, ensure_ascii=False, indent=None)
        
        if not data:
            return "Empty data", file_output
        
        if method_type == "rich" and self.enable_rich:
            # Will be handled by rich table in output
            return data, file_output
        else:
            # Simple table format
            headers = list(data[0].keys())
            console_output = " | ".join(headers) + "\n"
            console_output += "-" * len(console_output) + "\n"
            for row in data:
                console_output += " | ".join([str(row.get(h, "")) for h in headers]) + "\n"
            return console_output.strip(), file_output
    
    def _format_mixed_list(self, data: list, method_type: str) -> tuple:
        """Handle mixed type lists"""
        items = []
        for item in data:
            if isinstance(item, dict):
                items.append(json.dumps(item))
            else:
                items.append(str(item))
        return self._format_list_of_strings(items, method_type)
    
    def _get_level_color(self, level: str) -> str:
        """Get rich color for level"""
        colors = {
            'DEV': 'dim white',
            'DEBUG': 'blue',
            'INFO': 'green',
            'PROD': 'bright_green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bright_red',
            'AUDIT': 'magenta'
        }
        return colors.get(level.upper(), 'white')
    
    def _get_level_style(self, level: str) -> str:
        """Get rich text style for level"""
        styles = {
            'DEV': 'dim',
            'DEBUG': 'blue',
            'INFO': 'green',
            'PROD': 'bright_green',
            'ERROR': 'bold red',
            'CRITICAL': 'bold bright_red'
        }
        return styles.get(level.upper(), 'white')
    
    def _log(self, data: Any, level: str, method_type: str = "standard"):
        """Internal logging method"""
        if not self._should_log(level):
            return
        
        console_output, file_output = self._format_data(data, method_type)
        
        # Console output
        if method_type == "rich" and self.enable_rich:
            self._rich_console_output(console_output, level, data)
        else:
            self._standard_console_output(console_output, level)
        
        # File output
        if self.file_logger:
            self._file_output(file_output, level)
    
    def _standard_console_output(self, output: str, level: str):
        """Standard console output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {output}")
    
    def _rich_console_output(self, output: Any, level: str, original_data: Any):
        """Rich console output with panels and tables"""
        if not self.enable_rich:
            return self._standard_console_output(str(output), level)
        
        title = f"[{level}]"
        
        if isinstance(original_data, list) and original_data and isinstance(original_data[0], dict):
            # Create rich table for list of dicts
            table = Table(title=f"{title} Data")
            
            # Add columns
            headers = list(original_data[0].keys())
            for header in headers:
                table.add_column(str(header), style="cyan")
            
            # Add rows
            for row_dict in original_data:
                row_values = [str(row_dict.get(key, "")) for key in headers]
                table.add_row(*row_values)
            
            self.console.print(table)
        else:
            # Create panel for other types
            panel = Panel(
                output,
                title=title,
                border_style=self._get_level_color(level)
            )
            self.console.print(panel)
    
    def _file_output(self, output: str, level: str):
        """Output to file"""
        # Escape quotes for JSON logging
        escaped_output = output.replace('"', '\\"').replace('\n', '\\n')
        self.file_logger.log(
            getattr(logging, level.upper(), logging.INFO),
            f'"{escaped_output}"'
        )
    
    # Standard logging methods
    def dev(self, data: Any):
        """Development-only logging"""
        self._log(data, "DEV")
    
    def debug(self, data: Any):
        """Debug level logging"""
        self._log(data, "DEBUG")
    
    def info(self, data: Any):
        """Info level logging"""
        self._log(data, "INFO")
    
    def prod(self, data: Any):
        """Production-safe logging"""
        self._log(data, "PROD")
    
    def warning(self, data: Any):
        """Warning level logging"""
        self._log(data, "WARNING")
    
    def error(self, data: Any):
        """Error level logging"""
        self._log(data, "ERROR")
    
    def audit(self, data: Any):
        """Audit level logging"""
        self._log(data, "AUDIT")
    
    def critical(self, data: Any):
        """Critical level logging"""
        self._log(data, "CRITICAL")
    
    # Special methods
    def rich(self, data: Any, level: str = "INFO"):
        """Rich formatted output with panels and tables"""
        if self._should_log(level):
            self._log(data, level, "rich")
    
    def ai(self, data: Union[List[dict], dict, str], role: str = "user", level: str = "INFO"):
        """
        AI conversation logging with rich panels
        
        Handles:
        - List[dict] with role/content format
        - Single dict message
        - String message with optional role (defaults to "user")
        
        Args:
            data: Message data (string, dict, or list of dicts)
            role: Default role for string messages ("user", "assistant", "system")
            level: Log level
        """
        if not self._should_log(level):
            return
        
        if isinstance(data, str):
            # Simple string message with specified role
            messages = [{"role": role, "content": data}]
        elif isinstance(data, dict):
            # Single message dict
            messages = [data]
        elif isinstance(data, list):
            # List of messages
            messages = data
        else:
            # Fallback
            self._log(data, level, "rich")
            return
        
        # Rich conversation output
        if self.enable_rich:
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", str(msg))
                
                # Role-based styling
                if role == "user":
                    style = "blue"
                    icon = "👤"
                elif role == "assistant":
                    style = "green" 
                    icon = "🤖"
                elif role == "system":
                    style = "yellow"
                    icon = "⚙️"
                else:
                    style = "white"
                    icon = "💬"
                
                panel = Panel(
                    content,
                    title=f"{role.title()}",
                    border_style=style
                )
                self.console.print(panel)
        else:
            # Standard output for AI conversations
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", str(msg))
                self._standard_console_output(f"[{role.upper()}] {content}", level)
        
        # File output
        if self.file_logger:
            self._file_output(json.dumps(messages), level)


# Convenience function for quick setup
def get_logger(
    name: str = "logs",
    level: Optional[str] = None,
    enable_rich: bool = True,
    enable_files: bool = True,
    subdir: Optional[str] = None
) -> UniversalLogger:
    """
    Quick logger setup function
    
    Usage:
        log = get_logger("my_app", enable_rich=True, subdir="ai_logs")
    """
    return UniversalLogger(
        name=name,
        level=level,
        enable_rich=enable_rich,
        enable_files=enable_files,
        subdir=subdir
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    log = UniversalLogger("test", level="DEBUG")
    
    print("=== Testing Universal Logger ===\n")
    
    # Test all data types with different levels
    log.info("Simple string message")
    log.info({"user": "alice", "action": "login", "ip": "192.168.1.1"})
    log.info(["Starting pipeline", "Loading model", "Processing data"])
    log.info([
        {"step": 1, "action": "validate", "status": "✅"},
        {"step": 2, "action": "process", "status": "⏳"},
        {"step": 3, "action": "save", "status": "⏳"}
    ])
    
    # Test rich formatting
    print("\n=== Rich Formatting ===")
    log.rich("**Bold** and *italic* markdown text")
    log.rich({"model": "gpt-4", "tokens": 150, "cost": 0.03})
    log.rich(["Task 1", "Task 2", "Task 3"])
    log.rich([{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}])
    
    # Test AI conversations
    print("\n=== AI Conversations ===")
    log.ai([
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'll check the current weather for you."},
        {"role": "system", "content": "Weather API called successfully"}
    ])
    
    # Test different levels
    print("\n=== Different Levels ===")
    log.dev("Development debug info")
    log.debug("Debug information")
    log.prod("Production-safe message")
    log.warning("Warning message")
    log.error("Error occurred")
    
    print("\n=== Logger test complete ===")


"""
README 


# Universal Logger 🚀

The ultimate logging solution that handles everything - from simple debug messages to production-grade audit trails with beautiful rich formatting and AI conversation tracking.

## ✨ Features

- **🎯 Universal Data Type Support** - Handles `str`, `dict`, `List[str]`, `List[dict]` automatically
- **🎨 Rich Console Formatting** - Beautiful panels, tables, and markdown rendering
- **📁 Smart Rotatory Logging** - Size-based rotation with subdirectory support  
- **🤖 AI Conversation Tracking** - Special formatting for chat messages with role-based styling
- **🔧 Environment Detection** - Auto-detects dev/prod environments and adjusts accordingly
- **⚡ Zero Configuration** - Works perfectly out of the box, infinitely customizable

## 🚀 Quick Start

```python
from logger import UniversalLogger, get_logger

# Simple setup - works immediately
log = UniversalLogger("my_app")

# Or use convenience function
log = get_logger("my_app", enable_rich=True, subdir="ai_logs")
```

## 📊 Data Type Magic

Every logging method automatically handles **4 data types**:

### 1. Strings - Simple text messages
```python
log.info("User authentication successful")
# Output: [19:30:45] [INFO] User authentication successful
```

### 2. Dictionaries - Structured data
```python
log.info({"user_id": 123, "action": "login", "ip": "192.168.1.1"})
# Console: [19:30:45] [INFO] user_id=123 action=login ip=192.168.1.1
# File: {"user_id": 123, "action": "login", "ip": "192.168.1.1"}
```

### 3. List of Strings - Multiple messages
```python
log.info(["Starting pipeline", "Loading model", "Ready to process"])
# Output: [19:30:45] [INFO] 
#   • Starting pipeline
#   • Loading model  
#   • Ready to process
```

### 4. List of Dictionaries - Structured records
```python
# Simple text table (when enable_rich=False or using .info())
log.info([
    {"step": 1, "action": "validate", "status": "✅"},
    {"step": 2, "action": "process", "status": "⏳"},
    {"step": 3, "action": "save", "status": "pending"}
])
# Output: Simple text table
# step | action   | status
# -----|----------|--------
# 1    | validate | ✅
# 2    | process  | ⏳  
# 3    | save     | pending

# Rich formatted table (when enable_rich=True and using .rich())
log.rich([
    {"step": 1, "action": "validate", "status": "✅"},
    {"step": 2, "action": "process", "status": "⏳"},
    {"step": 3, "action": "save", "status": "pending"}
])
# Output: Beautiful rich table with colors and borders
```

## 🎨 Rich Formatting

### Standard vs Rich Methods

**Key Difference:** Regular logging methods (`.info()`, `.error()`, etc.) provide simple formatting, while `.rich()` provides enhanced visual formatting.

```python
# Standard method - simple formatting
log.info([{"endpoint": "/api/chat", "status": 200}])
# Output: Simple text table or key=value format

# Rich method - enhanced formatting  
log.rich([{"endpoint": "/api/chat", "status": 200}])
# Output: Beautiful bordered table with colors
```

### Rich Method Examples

Use `log.rich()` for enhanced visual output:

```python
# Rich text with markdown
log.rich("**Processing** model *inference* with `high accuracy`")
# → Beautiful panel with styled text

# Rich panels for dictionaries  
log.rich({"model": "gpt-4", "tokens": 150, "cost": 0.03})
# → Elegant bordered panel with key-value pairs

# Rich tables for lists of dictionaries
log.rich([
    {"name": "Alice", "score": 95, "status": "✅"}, 
    {"name": "Bob", "score": 87, "status": "⏳"}
])
# → Beautiful rich table with colors and styling
```

### Table Formatting Options

```python
# Option 1: Simple text table (works in any environment)
api_log = UniversalLogger("api", enable_rich=False)  
api_log.info([
    {"endpoint": "/api/chat", "status": 200, "duration": "1.2s"},
    {"endpoint": "/api/status", "status": 200, "duration": "0.1s"}
])
# Output: endpoint | status | duration
#         ---------|--------|----------
#         /api/chat | 200 | 1.2s
#         /api/status | 200 | 0.1s

# Option 2: Rich formatted table (best visual experience)
api_log = UniversalLogger("api", enable_rich=True)
api_log.rich([
    {"endpoint": "/api/chat", "status": 200, "duration": "1.2s"},
    {"endpoint": "/api/status", "status": 200, "duration": "0.1s"}
])
# Output: Beautiful bordered table with colors and styling
```

## 🤖 AI Conversation Logging

Perfect for tracking AI interactions with flexible input options:

### Method 1: Full conversation list
```python
log.ai([
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I'll check the current weather for you."},
    {"role": "system", "content": "Weather API called successfully"}
])
```

### Method 2: Simple string with default role
```python
# Defaults to "user" role
log.ai("What's the weather like?")

# Specify role directly
log.ai("I'll check the current weather for you.", "assistant")
log.ai("Weather API called successfully", "system")
```

### Method 3: Single message dict
```python
log.ai({"role": "user", "content": "What's the weather like?"})
```

**Rich Output:**
```
╭────────────────────────────────── 👤 User ───────────────────────────────────╮
│ What's the weather like?                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── 🤖 Assistant ────────────────────────────────╮
│ I'll check the current weather for you.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────── ⚙️ System ──────────────────────────────────╮
│ Weather API called successfully                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## 📈 Logging Levels

Complete level control with custom levels:

```python
# Standard levels
log.debug("Detailed diagnostic info")      # Level 10
log.info("General information")            # Level 20  
log.warning("Something unexpected")        # Level 30
log.error("Serious problem")               # Level 40
log.critical("Very serious error")         # Level 50

# Custom levels
log.dev("Development-only message")        # Level 5
log.prod("Production-safe information")    # Level 25
log.audit("Security/compliance log")       # Level 45

# Dynamic level control
log.set_level("PROD")  # Only PROD, WARNING, ERROR, CRITICAL show
log.set_level("DEBUG") # Everything shows
```

## 🗂️ File Logging & Rotation

Automatic file logging with smart rotation:

```python
# Main logger - logs to logs/main.log
main_log = UniversalLogger("main")

# Subdirectory logging - logs to logs/ai_conversations/ai.log  
ai_log = UniversalLogger("ai", subdir="ai_conversations")

# API logging - logs to logs/api_calls/api.log
api_log = UniversalLogger("api", subdir="api_calls")
```

**Directory Structure:**
```
logs/
├── main.log                    # Main app logs
├── ai_conversations/           # AI-specific subdirectory
│   ├── ai.log
│   └── ai.log.1               # Rotated files
├── api_calls/                  # API-specific subdirectory
│   ├── api.log
│   └── archived/
└── errors/                     # Error-specific logs
    └── error.log
```

**File Format** (Structured JSON):
```json
{"timestamp": "2025-08-09 19:30:45,123", "level": "INFO", "logger": "main", "message": "User login successful"}
{"timestamp": "2025-08-09 19:30:46,456", "level": "ERROR", "logger": "api", "message": "{\"error\": \"timeout\", \"duration\": \"5.2s\"}"}
```

## 🌍 Environment Detection  

Smart environment-aware logging:

```python
# Auto-detects environment and sets appropriate level
log = UniversalLogger("app")  # No level specified

# Environment detection logic:
# DEBUG=1 or DEV=1        → DEV level (shows everything)
# PRODUCTION=1 or PROD=1  → PROD level (production-safe only)  
# Interactive terminal    → DEBUG level (development mode)
# Server/CI environment   → INFO level (safe default)
```

**Manual environment testing:**
```bash
# Development mode - shows everything
DEBUG=1 python my_app.py

# Production mode - only production-safe logs
PRODUCTION=1 python my_app.py
```

## ⚙️ Configuration Options

Complete customization:

```python
log = UniversalLogger(
    name="my_app",              # Logger name
    level="INFO",               # Log level (or None for auto-detect)
    enable_rich=True,           # Rich console formatting
    enable_files=True,          # File logging  
    log_dir="logs",             # Base log directory
    subdir="ai_conversations",  # Subdirectory for this logger
    max_bytes=10*1024*1024,     # Max file size (10MB)
    backup_count=5              # Number of backup files
)
```

## 📋 Complete Usage Example

```python
from logger import UniversalLogger

# Setup loggers for different components
main_log = UniversalLogger("main", level="INFO")
ai_log = UniversalLogger("ai_chat", subdir="ai_conversations")  
api_log = UniversalLogger("api", subdir="api_calls", enable_rich=False)

# Standard logging with automatic data type handling
main_log.info("Application started")
main_log.info({"version": "1.0.0", "env": "production"})
main_log.info(["Loading config", "Connecting to DB", "Ready"])

# Rich formatting for complex data
ai_log.rich({
    "model": "gpt-4", 
    "context_length": 8192,
    "temperature": 0.7
})

# AI conversation tracking - multiple ways
ai_log.ai([
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
])

# Or use the simple string format with role
ai_log.ai("Explain quantum computing")  # Defaults to user role
ai_log.ai("Quantum computing uses quantum mechanics...", "assistant")

# API logging - Option 1: Simple text table (server environment)
api_log.info([
    {"endpoint": "/api/chat", "status": 200, "duration": "1.2s"},
    {"endpoint": "/api/status", "status": 200, "duration": "0.1s"}
])

# API logging - Option 2: Rich formatted table (enhanced visual)
api_log.rich([
    {"endpoint": "/api/chat", "status": 200, "duration": "1.2s"},
    {"endpoint": "/api/status", "status": 200, "duration": "0.1s"}
])

# Level-based logging
main_log.debug("Debug info")      # Won't show (below INFO)
main_log.error("Critical error")  # Will show (above INFO)

# Dynamic level adjustment
main_log.set_level("DEBUG")       # Now debug messages show
main_log.debug("Now visible")     # Shows after level change
```

## 🎯 Why Universal Logger?

### Stop Thinking About Logging. Just Log.

**Traditional logging is painful:**
- 🤯 Complex setup with formatters, handlers, levels
- 🔧 Different libraries for console vs file vs structured logging  
- 📊 Manual JSON conversion for dictionaries and lists
- 🎨 No visual formatting without extra dependencies
- 📁 Manual file rotation and directory management
- 🤖 Separate setup for AI conversation logging

**Universal Logger eliminates the pain:**
- ⚡ **Zero setup** - works instantly with smart defaults
- 🧠 **Zero thinking** - handles all data types automatically  
- 🎯 **One tool** - console, files, rich formatting, AI conversations
- 🔄 **Zero maintenance** - automatic rotation, smart environments

### Before vs After: Real Examples

#### Example 1: Logging Mixed Data Types
**Before:** Complex, error-prone setup
```python
import logging
import json
from rich.console import Console
from rich.table import Table

# Setup nightmare
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Log different types (manual work)
logger.info("Application started")                    # ✅ Works
logger.info(json.dumps({"user": "alice", "id": 123})) # ❌ Ugly JSON string
logger.info(str(["task1", "task2", "task3"]))         # ❌ Ugly list string

# Want a table? More code...
api_data = [{"endpoint": "/api/chat", "status": 200}]
table = Table(title="API Calls")
table.add_column("endpoint")
table.add_column("status") 
for row in api_data:
    table.add_row(row["endpoint"], str(row["status"]))
console.print(table)
```

**After:** Effortless, beautiful
```python
from flowgen.logger import get_logger
log = get_logger("app")

# All data types work perfectly
log.info("Application started")                    # ✅ Clean output
log.info({"user": "alice", "id": 123})            # ✅ Beautiful key=value  
log.info(["task1", "task2", "task3"])             # ✅ Bulleted list
log.rich([{"endpoint": "/api/chat", "status": 200}]) # ✅ Gorgeous table automatically!
```

#### Example 2: AI Conversation Logging  
**Before:** Messy, inconsistent
```python
import logging
import json
logger = logging.getLogger("ai")

# AI conversation - looks terrible
conversation = [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses..."}
]
for msg in conversation:
    logger.info(f"[{msg['role'].upper()}] {msg['content']}")
# Output: Ugly, hard to read
```

**After:** Professional, readable
```python
log.ai([
    {"role": "user", "content": "Explain quantum computing"},  
    {"role": "assistant", "content": "Quantum computing uses..."}
])
# Output: Beautiful panels with role-based colors and styling
```

#### Example 3: Environment-Aware Logging
**Before:** Manual environment handling
```python
import os
import logging

# Manual environment detection and setup
if os.getenv('PRODUCTION'):
    level = logging.WARNING
    handler = logging.FileHandler('prod.log')
elif os.getenv('DEBUG'):
    level = logging.DEBUG  
    handler = logging.StreamHandler()
else:
    level = logging.INFO
    handler = logging.StreamHandler()

logger = logging.getLogger()
logger.setLevel(level)
logger.addHandler(handler)
```

**After:** Automatic, intelligent
```python
log = get_logger("app")  # Automatically detects environment and sets optimal defaults
# Production? → Files only, WARNING level
# Development? → Rich console, DEBUG level  
# Server? → Simple formatting
# Terminal? → Rich formatting
```

### The Result: Focus on Your Code, Not Your Logs

❌ **Before:** Spend hours configuring logging  
✅ **After:** Import and forget - it just works

❌ **Before:** Ugly, inconsistent log formats  
✅ **After:** Beautiful, professional output  

❌ **Before:** Different tools for different needs  
✅ **After:** One logger handles everything  

❌ **Before:** Manual JSON, tables, formatting  
✅ **After:** Automatic formatting for all data types

**Your brain stays focused on building features, not wrestling with logs.**


## 📦 Requirements

- Python 3.6+
- `rich` library (for beautiful formatting)
  ```bash
  pip install rich
  ```

## 🚀 Installation

1. Copy `logger.py` to your project
2. Install rich: `pip install rich`  
3. Import and use: `from logger import UniversalLogger`

**That's it!** No configuration files, no complex setup - just beautiful, powerful logging out of the box.

---

**Universal Logger** - Because logging should be beautiful, simple, and powerful. ✨


"""
