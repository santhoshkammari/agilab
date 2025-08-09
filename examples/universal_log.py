from flowgen.logger import UniversalLogger

# Setup loggers for different components
main_log = UniversalLogger("main", level="INFO")
ai_log = UniversalLogger("ai_chat", subdir="ai_conversations")  
api_log = UniversalLogger("api", subdir="api_calls", enable_rich=True)

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

# Different ways to log AI messages
ai_log.ai({"role": "user", "content": "Explain quantum computing"})
ai_log.ai("Explain quantum computing")  # Defaults to user role
ai_log.ai("Quantum computing uses quantum mechanics...", "assistant")

# AI conversation with tool calling and results
ai_log.ai([
    {"role": "user", "content": "What's the weather in New York?"},
    {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"New York\"}"}}
    ]},
    {"role": "tool", "content": "{\"temperature\": 72, \"condition\": \"sunny\", \"humidity\": 45}", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "The weather in New York is currently 72°F and sunny with 45% humidity."}
])

# Individual tool call logging
ai_log.ai("I need to check the current weather for you.", "assistant")
ai_log.ai("Tool called: get_weather(location='New York')", "system")
ai_log.ai("{\"temperature\": 72, \"condition\": \"sunny\", \"humidity\": 45}", "tool")
ai_log.ai("The weather in New York is currently 72°F and sunny.", "assistant")

# API logging with rich table formatting
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
