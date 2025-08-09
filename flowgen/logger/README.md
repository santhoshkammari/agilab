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

Perfect for tracking AI interactions:

```python
log.ai([
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I'll check the current weather for you."},
    {"role": "system", "content": "Weather API called successfully"}
])
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

# AI conversation tracking
ai_log.ai([
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
])

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

**Before:** Multiple logging libraries, complex setup, inconsistent formatting
```python
import logging
import json
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(json.dumps({"user": "alice"}))  # Manual JSON conversion
```

**After:** One import, handles everything
```python  
from logger import get_logger
log = get_logger("app")
log.info({"user": "alice"})  # Automatic formatting
log.rich(complex_data)       # Beautiful output  
log.ai(conversation)         # AI-specific formatting
```

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