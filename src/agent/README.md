# FlowGen Agent Framework: Complete Technical Guide

*Building Autonomous AI Agents with Any LLM*

---

## Table of Contents

1. [Introduction & Philosophy](#introduction--philosophy)
2. [Quick Start](#quick-start)
3. [Basic Agent Usage](#basic-agent-usage)
4. [Tool Integration](#tool-integration)
5. [History Management](#history-management)
6. [Agent Chaining](#agent-chaining)
7. [Serialization & Persistence](#serialization--persistence)
8. [Streaming & Real-time](#streaming--real-time)
9. [Advanced Patterns](#advanced-patterns)
10. [Production Best Practices](#production-best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Introduction & Philosophy

The FlowGen Agent framework transforms any LLM into an autonomous agent capable of tool usage, conversation continuity, and complex reasoning workflows. Built on the principle of **transparency and composability**, every agent operation is visible and controllable.

### Core Principles

- **LLM Agnostic**: Works with vLLM, Gemini, Ollama, or any BaseLLM
- **Drop-in Replacement**: Use `agent()` exactly like `llm()`
- **Full Transparency**: Every tool call, response, and decision is visible
- **Composable**: Chain agents, persist state, share conversations
- **Production Ready**: Built for scale with streaming, batching, and error handling

---

## Quick Start

### Installation & Setup

First, ensure you have the FlowGen framework installed:

```bash
# From the deepresearch project directory
cd src/deepresearch
python -c "from flowgen import Agent, Gemini; print('✅ FlowGen imported successfully')"
```

```python
from flowgen import Agent, Gemini, vLLM

# Create any LLM - requires API keys/endpoints
llm = Gemini(api_key="your-api-key")  
# or llm = vLLM(host="localhost", port="8000", model="your-model")
# or llm = Ollama(host="localhost", port="11434", model="llama2")

# Wrap with Agent - no additional setup needed
agent = Agent(llm)

# Use exactly like an LLM
result = agent("What's 2+2?")
print(result['content'])  # "The answer is 4"
print(result['iterations'])  # 1 (no tools needed)
```

### Your First Tool-Using Agent

```python
def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))  # Note: Use safe math parser in production
    except:
        return "Invalid expression"

# Agent with tools
agent = Agent(llm, tools=[calculate])

result = agent("What's 15 * 23 + 100?")
# Agent automatically calls calculate() and provides final answer

print(result['content'])     # "The result is 445"
print(result['iterations']) # 2 (LLM call + tool call + final response)
print(len(result['messages']))  # Shows full conversation with tool calls
```

---

## Basic Agent Usage

### 1. Creating Agents

```python
from flowgen import Agent, Gemini

llm = Gemini()

# Basic agent
basic_agent = Agent(llm)

# Agent with configuration
configured_agent = Agent(
    llm=llm,
    tools=[func1, func2],
    max_iterations=5,
    stream=False
)
```

### 2. Input Flexibility

The Agent accepts the same inputs as any LLM:

```python
# String input
result = agent("Hello world")

# Message format
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What's the weather?"}
]
result = agent(messages)

# Batch processing
queries = ["What's 2+2?", "What's 3+3?"]
results = agent(queries)  # Returns list of results
```

### 3. Understanding Agent Responses

```python
result = agent("Calculate 5*7")

print(result['content'])     # Final response text
print(result['think'])       # Reasoning (if supported by LLM)
print(result['iterations'])  # Number of tool-calling rounds
print(result['final'])       # True when complete
```

---

## Tool Integration

### 1. Creating Tools

Tools are simple Python functions with docstrings:

```python
def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: City name or coordinates
    """
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 25°C"

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.
    
    Args:
        query: Search terms
        max_results: Maximum number of results to return
    """
    # Your search logic here
    return f"Search results for '{query}': [Results...]"
```

### 2. Tool Function Signatures

The framework automatically converts Python functions to LLM tools:

```python
def complex_tool(
    text: str,           # Required string
    count: int = 10,     # Optional with default
    options: list = None, # Optional list
    flag: bool = False   # Optional boolean
) -> dict:
    """Complex tool with multiple parameter types.
    
    Args:
        text: Input text to process
        count: Number of items to process
        options: List of processing options
        flag: Enable special processing
    """
    return {
        "processed": text.upper(),
        "count": count,
        "flag_used": flag
    }
```

### 3. Dynamic Tool Management

```python
agent = Agent(llm, tools=[tool1, tool2])

# Add tools dynamically
agent.add_tool(new_tool)

# Remove tools
agent.remove_tool('tool_name')

# Tools can also be provided at call time
result = agent("Query", tools=[special_tool])
```

### 4. Tool Execution Flow

```python
def debug_tool(message: str) -> str:
    """Debug tool that shows execution."""
    print(f"🔧 Tool called with: {message}")
    result = f"Processed: {message.upper()}"
    print(f"✅ Tool result: {result}")
    return result

agent = Agent(llm, tools=[debug_tool])
result = agent("Process 'hello world'")

# Output:
# 🔧 Tool called with: hello world
# ✅ Tool result: Processed: HELLO WORLD
```

---

## History Management

### 1. Agent with Initial History

```python
# Previous conversation context
history = [
    {"role": "system", "content": "You are a math tutor"},
    {"role": "user", "content": "I'm learning algebra"},
    {"role": "assistant", "content": "Great! I'll help with algebra problems."}
]

# Create agent with history
agent = Agent(llm, tools=[calculate], history=history)

# New query continues the conversation
result = agent("What's a quadratic equation?")
# Agent knows it's tutoring algebra to the user
```

### 2. Dynamic History Management

```python
agent = Agent(llm, tools=[tools])

# Add system context
agent.add_history([
    {"role": "system", "content": "You are an expert researcher"}
])

# Add conversation snippets
agent.add_history([
    {"role": "user", "content": "I'm researching climate change"},
    {"role": "assistant", "content": "I'll help you find reliable sources"}
])

# Query with full context
result = agent("Find recent papers on renewable energy")
```

### 3. History Inspection & Management

```python
# Get current conversation
conversation = agent.get_conversation()
print(f"Conversation has {len(conversation)} messages")

# Replace history entirely
new_history = load_conversation_from_file()
agent.set_history(new_history)

# Clear everything
agent.clear_history()
```

### 4. Cross-Agent History Sharing

```python
# Agent 1 does research
research_agent = Agent(llm, tools=[search_web])
research_result = research_agent("Research Python frameworks")

# Agent 2 continues with research context
analysis_agent = Agent(
    llm, 
    tools=[analyze_data],
    history=research_agent.get_conversation()
)
analysis_result = analysis_agent("Analyze the frameworks you found")
```

---

## Agent Chaining

### 1. Basic Chaining with `>>` Operator

```python
# Create specialized agents
research_agent = Agent(llm, tools=[search_web, get_papers])
analysis_agent = Agent(llm, tools=[analyze_data, create_chart])
summary_agent = Agent(llm, tools=[])

# Chain them together
pipeline = research_agent >> analysis_agent >> summary_agent

# Execute the entire pipeline
result = pipeline("Research renewable energy trends and analyze")

print(result['content'])        # Final summary
print(result['chain_length'])   # 3
print(result['chain_results'])  # All intermediate results
```

### 2. Understanding Chain Execution

```python
def log_tool(message: str) -> str:
    print(f"Tool executed: {message}")
    return f"Logged: {message}"

agent1 = Agent(llm, tools=[log_tool])
agent2 = Agent(llm, tools=[log_tool])
agent3 = Agent(llm, tools=[log_tool])

chain = agent1 >> agent2 >> agent3
result = chain("Process this message")

# Shows execution flow:
# Agent 1: processes "Process this message"
# Agent 2: processes Agent 1's output
# Agent 3: processes Agent 2's output
```

### 3. Conditional Chaining

```python
def route_request(request_type: str) -> str:
    """Route requests to appropriate handlers."""
    if request_type == "math":
        return "Route to math specialist"
    elif request_type == "research":
        return "Route to research specialist"
    else:
        return "Route to general assistant"

router_agent = Agent(llm, tools=[route_request])
math_agent = Agent(llm, tools=[calculate])
research_agent = Agent(llm, tools=[search_web])

# Dynamic routing based on first agent's decision
first_result = router_agent("I need help with calculus")
if "math" in first_result['content'].lower():
    final_result = math_agent("Solve derivative of x^2")
```

### 4. Chain Serialization

```python
chain = agent1 >> agent2 >> agent3
result = chain("Query")

# Export entire chain conversation
chain_export = chain.export('json')

# Or get markdown of full pipeline
chain_md = chain.export('markdown')
print(chain_md)  # Shows all 3 agent conversations
```

---

## Serialization & Persistence

### 1. Basic Export/Import

```python
# Create and use agent
agent = Agent(llm, tools=[calculate, search])
result = agent("Calculate 15*23 and search for Python tutorials")

# Export conversation
json_data = agent.export('json')

# Later, restore the agent
restored_agent = Agent.load(
    data=json_data,
    llm=new_llm_instance,
    tools=[calculate, search]  # Must provide same tools
)

# Continue conversation
continue_result = restored_agent("Now calculate 45/9")
```

### 2. Export Formats

```python
agent = Agent(llm, tools=[get_weather])
result = agent("What's the weather in Tokyo?")

# Dictionary format (for programmatic use)
dict_export = agent.export('dict')
print(dict_export['conversation'])
print(dict_export['tools'])

# JSON format (for storage)
json_export = agent.export('json')
with open('agent_state.json', 'w') as f:
    f.write(json_export)

# Markdown format (for human reading / other agents)
md_export = agent.export('markdown')
print(md_export)
```

### 3. Markdown Export Example

```markdown
# Agent Conversation Export

**Model:** gemini-2.5-flash
**Tools:** get_weather, calculate
**Max Iterations:** 10

## Conversation

**👤 User:** What's the weather in Tokyo and what's 2+2?

**🤖 Assistant:** *Called tools: get_weather, calculate*

**⚡ Tool (get_weather):** Weather in Tokyo: Sunny, 25°C

**⚡ Tool (calculate):** 4

**🤖 Assistant:** The weather in Tokyo is sunny and 25°C. Also, 2+2 equals 4.
```

### 4. Cross-Session Persistence

```python
# Session 1: Research phase
research_agent = Agent(llm, tools=[search_web])
research_result = research_agent("Research machine learning trends")

# Save state
with open('research_session.json', 'w') as f:
    f.write(research_agent.export('json'))

# Session 2: Analysis phase (later/different process)
with open('research_session.json', 'r') as f:
    saved_state = f.read()

analysis_agent = Agent.load(
    data=saved_state,
    llm=llm,
    tools=[search_web, analyze_data]  # Can add new tools
)

# Continue where we left off
analysis_result = analysis_agent("Now analyze the trends you found")
```

### 5. Database Integration Pattern

```python
import json
import sqlite3

class AgentPersistence:
    def __init__(self, db_path: str = "agents.db"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id TEXT PRIMARY KEY,
                agent_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def save_agent(self, session_id: str, agent: Agent):
        data = agent.export('json')
        self.db.execute(
            "INSERT OR REPLACE INTO agent_sessions (id, agent_data) VALUES (?, ?)",
            (session_id, data)
        )
        self.db.commit()
    
    def load_agent(self, session_id: str, llm, tools) -> Agent:
        cursor = self.db.execute(
            "SELECT agent_data FROM agent_sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        
        return Agent.load(row[0], llm, tools)

# Usage
persistence = AgentPersistence()

# Save agent state
agent = Agent(llm, tools=[calculate])
result = agent("What's 15*23?")
persistence.save_agent("user_123_math", agent)

# Load agent state
restored_agent = persistence.load_agent("user_123_math", llm, [calculate])
```

---

## Streaming & Real-time

### 1. Basic Streaming

```python
# Create streaming agent
stream_agent = Agent(llm, tools=[search_web], stream=True)

# Process with real-time feedback
for event in stream_agent("Search for Python tutorials"):
    if event['type'] == 'iteration_start':
        print(f"🔄 Starting iteration {event['iteration']}")
    
    elif event['type'] == 'llm_response':
        print(f"🤖 LLM: {event['content'][:50]}...")
    
    elif event['type'] == 'tool_start':
        print(f"🔧 Calling {event['tool_name']}")
    
    elif event['type'] == 'tool_result':
        print(f"✅ Result: {event['result'][:50]}...")
    
    elif event['type'] == 'final':
        print(f"🏁 Final: {event['content']}")
        break
```

### 2. Event Types

```python
stream_agent = Agent(llm, tools=[calculate, search], stream=True)

for event in stream_agent("Calculate 15*23 then search for math tutorials"):
    event_type = event['type']
    
    if event_type == 'iteration_start':
        # New reasoning iteration beginning
        print(f"Iteration {event['iteration']} started")
    
    elif event_type == 'llm_response':
        # LLM provided response (may include tool calls)
        print(f"LLM thinking: {event['think']}")
        print(f"LLM response: {event['content']}")
        print(f"Tools to call: {event['tools']}")
    
    elif event_type == 'tool_start':
        # About to execute a tool
        print(f"Executing {event['tool_name']} with {event['tool_args']}")
    
    elif event_type == 'tool_result':
        # Tool execution completed
        print(f"Tool {event['tool_name']} returned: {event['result']}")
    
    elif event_type == 'final':
        # Agent finished (no more tool calls)
        print(f"Final answer: {event['content']}")
        if event.get('truncated'):
            print("Warning: Max iterations reached")
        break
```

### 3. Web UI Integration

```python
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('agent_query')
def handle_query(data):
    query = data['query']
    agent = Agent(llm, tools=[search_web, calculate], stream=True)
    
    for event in agent(query):
        # Send real-time updates to frontend
        emit('agent_event', {
            'type': event['type'],
            'content': event.get('content', ''),
            'tool_name': event.get('tool_name', ''),
            'iteration': event.get('iteration', 0)
        })
        
        if event['type'] == 'final':
            emit('agent_complete', {'result': event['content']})
            break
```

### 4. Progress Tracking

```python
class ProgressTracker:
    def __init__(self):
        self.events = []
        self.current_iteration = 0
        self.total_tools_called = 0
    
    def track_agent(self, agent_stream):
        for event in agent_stream:
            self.events.append(event)
            
            if event['type'] == 'iteration_start':
                self.current_iteration = event['iteration']
                print(f"Progress: Iteration {self.current_iteration}")
            
            elif event['type'] == 'tool_result':
                self.total_tools_called += 1
                print(f"Progress: {self.total_tools_called} tools executed")
            
            elif event['type'] == 'final':
                print(f"Complete: {self.current_iteration} iterations, {self.total_tools_called} tools")
                return event
        
        return None

# Usage
tracker = ProgressTracker()
agent = Agent(llm, tools=[complex_tool1, complex_tool2], stream=True)
final_result = tracker.track_agent(agent("Complex multi-step query"))
```

---

## Advanced Patterns

### 1. Multi-Agent Collaboration

```python
# Specialized agents
data_agent = Agent(llm, tools=[fetch_data, clean_data])
analysis_agent = Agent(llm, tools=[analyze_patterns, create_visualizations])
report_agent = Agent(llm, tools=[generate_report, format_document])

def collaborative_analysis(query: str):
    # Step 1: Data collection
    print("🔍 Data collection phase...")
    data_result = data_agent(f"Collect data for: {query}")
    
    # Step 2: Analysis with data context
    print("📊 Analysis phase...")
    analysis_agent.add_history(data_agent.get_conversation())
    analysis_result = analysis_agent("Analyze the collected data")
    
    # Step 3: Report generation with full context
    print("📝 Report generation phase...")
    report_agent.add_history(data_agent.get_conversation())
    report_agent.add_history(analysis_agent.get_conversation())
    final_result = report_agent("Generate comprehensive report")
    
    return final_result

result = collaborative_analysis("Market trends in renewable energy")
```

### 2. Agent Hierarchies

```python
class SupervisorAgent:
    def __init__(self, llm, specialist_agents: dict):
        self.supervisor = Agent(llm, tools=[self.delegate_task])
        self.specialists = specialist_agents
    
    def delegate_task(self, task_type: str, task_description: str) -> str:
        """Delegate tasks to specialist agents."""
        if task_type in self.specialists:
            specialist = self.specialists[task_type]
            result = specialist(task_description)
            return f"Specialist result: {result['content']}"
        else:
            return f"No specialist available for {task_type}"
    
    def process(self, query: str):
        return self.supervisor(query)

# Create specialists
math_agent = Agent(llm, tools=[calculate, solve_equations])
research_agent = Agent(llm, tools=[search_web, get_papers])
code_agent = Agent(llm, tools=[run_code, debug_code])

# Create supervisor
supervisor = SupervisorAgent(llm, {
    'math': math_agent,
    'research': research_agent,
    'coding': code_agent
})

# Supervisor automatically routes tasks
result = supervisor.process("I need to solve x^2 + 5x + 6 = 0 and find research papers about quadratic equations")
```

### 3. Retry and Error Handling

```python
def robust_tool_call(func_name: str, max_retries: int = 3):
    """Create a tool with retry logic."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                result = original_tools[func_name](*args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error after {max_retries} attempts: {str(e)}"
                print(f"Attempt {attempt + 1} failed: {e}")
        return "Max retries exceeded"
    
    wrapper.__name__ = func_name
    wrapper.__doc__ = original_tools[func_name].__doc__
    return wrapper

# Create robust versions of tools
robust_search = robust_tool_call('search_web')
robust_calculate = robust_tool_call('calculate')

agent = Agent(llm, tools=[robust_search, robust_calculate])
```

### 4. Agent Memory Systems

```python
class MemoryAgent:
    def __init__(self, llm, tools, memory_file: str = "agent_memory.json"):
        self.memory_file = memory_file
        self.long_term_memory = self.load_memory()
        
        # Add memory tools
        memory_tools = [self.remember, self.recall, self.forget]
        all_tools = tools + memory_tools
        
        self.agent = Agent(llm, tools=all_tools)
    
    def remember(self, key: str, value: str) -> str:
        """Store information in long-term memory."""
        self.long_term_memory[key] = value
        self.save_memory()
        return f"Remembered: {key} = {value}"
    
    def recall(self, key: str) -> str:
        """Retrieve information from long-term memory."""
        value = self.long_term_memory.get(key, "Not found")
        return f"Recalled: {key} = {value}"
    
    def forget(self, key: str) -> str:
        """Remove information from long-term memory."""
        if key in self.long_term_memory:
            del self.long_term_memory[key]
            self.save_memory()
            return f"Forgot: {key}"
        return f"Key {key} not found in memory"
    
    def load_memory(self) -> dict:
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.long_term_memory, f, indent=2)
    
    def __call__(self, query: str):
        return self.agent(query)

# Usage
memory_agent = MemoryAgent(llm, tools=[calculate])

# Agent can now remember things across sessions
result1 = memory_agent("Remember that my favorite number is 42")
result2 = memory_agent("What's my favorite number times 2?")
# Agent recalls 42 and calculates 84
```

### 5. Dynamic Tool Loading

```python
class DynamicAgent:
    def __init__(self, llm, tool_directory: str = "tools/"):
        self.llm = llm
        self.tool_directory = tool_directory
        self.loaded_tools = {}
        
        # Core tools for tool management
        self.agent = Agent(llm, tools=[
            self.load_tool,
            self.list_available_tools,
            self.unload_tool
        ])
    
    def load_tool(self, tool_name: str) -> str:
        """Dynamically load a tool from the tools directory."""
        try:
            tool_path = f"{self.tool_directory}{tool_name}.py"
            spec = importlib.util.spec_from_file_location(tool_name, tool_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the main function from the module
            tool_func = getattr(module, tool_name)
            self.loaded_tools[tool_name] = tool_func
            
            # Add to agent
            self.agent.add_tool(tool_func)
            
            return f"Successfully loaded tool: {tool_name}"
        except Exception as e:
            return f"Failed to load tool {tool_name}: {str(e)}"
    
    def list_available_tools(self) -> str:
        """List all available tools in the tools directory."""
        import os
        tool_files = [f[:-3] for f in os.listdir(self.tool_directory) if f.endswith('.py')]
        loaded = list(self.loaded_tools.keys())
        return f"Available: {tool_files}, Loaded: {loaded}"
    
    def unload_tool(self, tool_name: str) -> str:
        """Unload a tool from the agent."""
        if tool_name in self.loaded_tools:
            self.agent.remove_tool(tool_name)
            del self.loaded_tools[tool_name]
            return f"Unloaded tool: {tool_name}"
        return f"Tool {tool_name} not loaded"
    
    def __call__(self, query: str):
        return self.agent(query)

# Usage
dynamic_agent = DynamicAgent(llm)

# Agent can load tools on demand
result = dynamic_agent("Load the weather_api tool and get weather for Paris")
```

---

## Production Best Practices

### 1. Error Handling & Validation

```python
class ProductionAgent:
    def __init__(self, llm, tools, max_iterations=5):
        self.agent = Agent(
            llm=llm,
            tools=self.validate_tools(tools),
            max_iterations=max_iterations
        )
    
    def validate_tools(self, tools):
        """Validate tools before adding to agent."""
        validated = []
        for tool in tools:
            if not callable(tool):
                print(f"Warning: {tool} is not callable, skipping")
                continue
            
            if not tool.__doc__:
                print(f"Warning: {tool.__name__} has no docstring")
            
            validated.append(tool)
        
        return validated
    
    def safe_call(self, query: str, timeout: int = 30):
        """Safe agent call with timeout and error handling."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Agent call timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = self.agent(query)
            signal.alarm(0)  # Cancel timeout
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except TimeoutError:
            return {
                'success': False,
                'result': None,
                'error': 'Timeout exceeded'
            }
        except Exception as e:
            signal.alarm(0)
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
```

### 2. Logging & Monitoring

```python
import logging
from datetime import datetime

class MonitoredAgent:
    def __init__(self, llm, tools, log_level=logging.INFO):
        self.logger = logging.getLogger(f"agent_{id(self)}")
        self.logger.setLevel(log_level)
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.agent = Agent(llm, tools=tools, stream=True)
        self.metrics = {
            'total_calls': 0,
            'total_tools_used': 0,
            'average_iterations': 0,
            'errors': []
        }
    
    def __call__(self, query: str):
        start_time = datetime.now()
        self.logger.info(f"Starting agent call: {query[:100]}")
        
        try:
            iterations = 0
            tools_used = 0
            
            for event in self.agent(query):
                if event['type'] == 'iteration_start':
                    iterations = event['iteration']
                    self.logger.debug(f"Iteration {iterations} started")
                
                elif event['type'] == 'tool_start':
                    tools_used += 1
                    self.logger.info(f"Calling tool: {event['tool_name']}")
                
                elif event['type'] == 'tool_result':
                    self.logger.debug(f"Tool result: {event['result'][:100]}")
                
                elif event['type'] == 'final':
                    duration = datetime.now() - start_time
                    
                    self.logger.info(f"Agent completed in {duration.total_seconds()}s")
                    self.logger.info(f"Used {tools_used} tools in {iterations} iterations")
                    
                    # Update metrics
                    self.metrics['total_calls'] += 1
                    self.metrics['total_tools_used'] += tools_used
                    self.metrics['average_iterations'] = (
                        self.metrics['average_iterations'] * (self.metrics['total_calls'] - 1) + iterations
                    ) / self.metrics['total_calls']
                    
                    return event
        
        except Exception as e:
            self.logger.error(f"Agent error: {str(e)}")
            self.metrics['errors'].append({
                'timestamp': datetime.now(),
                'error': str(e),
                'query': query[:100]
            })
            raise
    
    def get_metrics(self):
        return self.metrics.copy()
```

### 3. Rate Limiting & Resource Management

```python
import time
from collections import deque
from threading import Lock

class RateLimitedAgent:
    def __init__(self, llm, tools, max_calls_per_minute=60):
        self.agent = Agent(llm, tools=tools)
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = deque()
        self.lock = Lock()
    
    def _check_rate_limit(self):
        """Check if rate limit allows new call."""
        current_time = time.time()
        
        with self.lock:
            # Remove calls older than 1 minute
            while self.call_times and current_time - self.call_times[0] > 60:
                self.call_times.popleft()
            
            # Check if we can make a new call
            if len(self.call_times) >= self.max_calls_per_minute:
                sleep_time = 60 - (current_time - self.call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self._check_rate_limit()
            
            # Record this call
            self.call_times.append(current_time)
    
    def __call__(self, query: str):
        self._check_rate_limit()
        return self.agent(query)
```

### 4. Configuration Management

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    max_iterations: int = 10
    stream: bool = False
    timeout: int = 30
    log_level: str = "INFO"
    rate_limit: int = 60
    enable_metrics: bool = True
    memory_file: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            max_iterations=int(os.getenv('AGENT_MAX_ITERATIONS', 10)),
            stream=os.getenv('AGENT_STREAM', 'false').lower() == 'true',
            timeout=int(os.getenv('AGENT_TIMEOUT', 30)),
            log_level=os.getenv('AGENT_LOG_LEVEL', 'INFO'),
            rate_limit=int(os.getenv('AGENT_RATE_LIMIT', 60)),
            enable_metrics=os.getenv('AGENT_METRICS', 'true').lower() == 'true',
            memory_file=os.getenv('AGENT_MEMORY_FILE')
        )

class ConfigurableAgent:
    def __init__(self, llm, tools, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        self.agent = Agent(
            llm=llm,
            tools=tools,
            max_iterations=self.config.max_iterations,
            stream=self.config.stream
        )
        
        if self.config.enable_metrics:
            self.metrics = MonitoredAgent(llm, tools)
        
        if self.config.rate_limit > 0:
            self.rate_limiter = RateLimitedAgent(llm, tools, self.config.rate_limit)
    
    def __call__(self, query: str):
        if hasattr(self, 'rate_limiter'):
            return self.rate_limiter(query)
        elif hasattr(self, 'metrics'):
            return self.metrics(query)
        else:
            return self.agent(query)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Tool Not Found Errors

**Problem**: `Error: Tool 'function_name' not found`

**Solution**:
```python
# Ensure tool functions are properly defined
def my_tool(param: str) -> str:
    """Tool description."""
    return result

# Check tool names match function names
agent = Agent(llm, tools=[my_tool])  # Not ['my_tool']

# Debug available tools
print([tool.__name__ for tool in agent.tools])
```

#### 2. Serialization Errors

**Problem**: `ValueError: Missing tools for loading: {'tool_name'}`

**Solution**:
```python
# When loading, provide all tools that were in original agent
original_tools = [tool1, tool2, tool3]
agent = Agent(llm, tools=original_tools)

# Export
data = agent.export('json')

# Load with same tools
restored_agent = Agent.load(data, llm, tools=original_tools)
```

#### 3. Infinite Tool Calling

**Problem**: Agent keeps calling tools indefinitely

**Solutions**:
```python
# Set reasonable iteration limit
agent = Agent(llm, tools=tools, max_iterations=5)

# Check tool outputs are helpful
def better_tool(query: str) -> str:
    result = process_query(query)
    # Return clear, complete result
    return f"Query '{query}' processed. Result: {result}"

# Monitor with streaming
for event in Agent(llm, tools=tools, stream=True)(query):
    if event['type'] == 'iteration_start' and event['iteration'] > 3:
        print("Warning: Many iterations")
```

#### 4. Memory/Performance Issues

**Problem**: Agent uses too much memory or is slow

**Solutions**:
```python
# Clear history periodically
agent.clear_history()

# Use streaming for long conversations
agent = Agent(llm, tools=tools, stream=True)

# Limit conversation history
def trim_history(agent, max_messages=50):
    conversation = agent.get_conversation()
    if len(conversation) > max_messages:
        agent.set_history(conversation[-max_messages:])
```

#### 5. Tool Parameter Type Errors

**Problem**: Tools receive wrong parameter types

**Solution**:
```python
# Use proper type hints
def typed_tool(count: int, text: str, flag: bool = False) -> str:
    """Tool with proper types.
    
    Args:
        count: Number of items (integer)
        text: Input text string
        flag: Optional boolean flag
    """
    # Validate types in tool if needed
    if not isinstance(count, int):
        return f"Error: count must be integer, got {type(count)}"
    
    return f"Processed {count} items from '{text}'"
```

### Debug Mode

```python
class DebugAgent:
    def __init__(self, llm, tools):
        self.agent = Agent(llm, tools=tools, stream=True)
    
    def debug_call(self, query: str):
        print(f"🐛 DEBUG: Starting query: {query}")
        print(f"🐛 DEBUG: Available tools: {[t.__name__ for t in self.agent.tools]}")
        
        for event in self.agent(query):
            print(f"🐛 DEBUG: Event: {event['type']}")
            
            if event['type'] == 'llm_response':
                print(f"🐛 DEBUG: LLM wants to call: {[t['name'] for t in event.get('tools', [])]}")
            
            elif event['type'] == 'tool_start':
                print(f"🐛 DEBUG: Calling {event['tool_name']} with: {event['tool_args']}")
            
            elif event['type'] == 'tool_result':
                print(f"🐛 DEBUG: Tool returned: {event['result'][:100]}...")
            
            elif event['type'] == 'final':
                print(f"🐛 DEBUG: Final result: {event['content'][:100]}...")
                return event

# Usage
debug_agent = DebugAgent(llm, [problematic_tool])
result = debug_agent.debug_call("Test query")
```

---

## Real-World Examples

### Example 1: Research Assistant Agent

```python
def search_papers(query: str, max_results: int = 5) -> str:
    """Search for academic papers."""
    # Mock implementation - replace with real API
    return f"Found {max_results} papers about '{query}'"

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize text to key points."""
    # Mock implementation - replace with real summarization
    sentences = text.split('.')[:max_sentences]
    return '. '.join(sentences) + '.'

# Create research assistant
research_assistant = Agent(
    llm=Gemini(),
    tools=[search_papers, summarize_text],
    max_iterations=5
)

# Use for research workflow
result = research_assistant("""
Research recent papers on large language models and summarize 
the key findings about their reasoning capabilities.
""")

print(result['content'])
```

### Example 2: Data Analysis Pipeline

```python
def load_dataset(dataset_name: str) -> str:
    """Load a dataset by name."""
    return f"Loaded dataset: {dataset_name} with 1000 rows, 10 columns"

def analyze_data(analysis_type: str) -> str:
    """Perform data analysis."""
    analyses = {
        'summary': "Mean: 45.2, Std: 12.8, Min: 10, Max: 98",
        'correlation': "Strong correlation between age and income (r=0.74)",
        'trends': "Upward trend in last 5 years, seasonal patterns detected"
    }
    return analyses.get(analysis_type, "Analysis type not supported")

def create_visualization(chart_type: str, data_description: str) -> str:
    """Create data visualizations."""
    return f"Created {chart_type} chart showing {data_description}"

# Create data analysis agents
data_agent = Agent(llm, tools=[load_dataset, analyze_data])
viz_agent = Agent(llm, tools=[create_visualization])

# Chained analysis
analysis_pipeline = data_agent >> viz_agent

result = analysis_pipeline("""
Load the customer_data dataset, perform summary and correlation analysis,
then create appropriate visualizations for the findings.
""")
```

### Example 3: Code Review Agent

```python
def run_tests(test_command: str) -> str:
    """Run test command and return results."""
    # Mock - replace with actual test execution
    return "✅ All 15 tests passed. Coverage: 87%"

def check_code_style(file_path: str) -> str:
    """Check code style violations."""
    return f"Found 3 style issues in {file_path}: line too long (2), missing docstring (1)"

def suggest_improvements(code_snippet: str) -> str:
    """Suggest code improvements."""
    return "Suggestions: 1) Add type hints, 2) Extract complex logic to separate functions, 3) Add error handling"

# Code review agent with conversation history
code_reviewer = Agent(
    llm=llm,
    tools=[run_tests, check_code_style, suggest_improvements],
    history=[
        {"role": "system", "content": "You are an expert code reviewer focusing on Python best practices."}
    ]
)

# Persistent code review session
review_result = code_reviewer("Please review my new user authentication module")

# Continue the review
code_reviewer.add_history(code_reviewer.get_conversation())
security_review = code_reviewer("Now focus specifically on security vulnerabilities")
```

## Conclusion

The FlowGen Agent framework provides a powerful, flexible foundation for building autonomous AI agents. From simple tool-using agents to complex multi-agent systems, the framework scales with your needs while maintaining transparency and control.

### Key Takeaways

1. **Start Simple**: Begin with basic agents and gradually add complexity
2. **Tool Design**: Well-designed tools with clear docstrings are crucial
3. **History Management**: Use history for context continuity across conversations
4. **Chaining**: Compose complex workflows from specialized agents
5. **Persistence**: Serialize and restore agent state for long-running applications
6. **Streaming**: Use streaming for real-time feedback in interactive applications
7. **Production**: Implement proper error handling, logging, and monitoring

### Architecture Benefits

- **Modular**: Each component (LLM, Agent, Tools) is independently replaceable
- **Testable**: Tools are pure functions, agents have predictable interfaces
- **Scalable**: From single agents to multi-agent orchestration
- **Observable**: Full visibility into agent decision-making process
- **Portable**: Export/import agent state across systems and sessions

### Performance Characteristics

- **Memory Efficient**: Conversation history management with cleanup options
- **Network Optimized**: Batching support for multiple queries
- **Error Resilient**: Graceful handling of tool failures and timeouts
- **Rate Limit Aware**: Built-in support for API rate limiting

### Next Steps

- **Experiment**: Try different LLM providers (vLLM, Gemini, Ollama)
- **Build Tools**: Create domain-specific tool libraries for your use case
- **Templates**: Develop agent templates for common patterns
- **Advanced Patterns**: Implement hierarchical agents and complex workflows
- **Production**: Add monitoring, logging, and error handling for deployment
- **Community**: Share tools and patterns with the FlowGen ecosystem

### Resources

- **Source Code**: `/src/deepresearch/flowgen/`
- **Examples**: `/src/deepresearch/tutorial/`
- **Tests**: Run `python -m pytest` for test suite
- **Documentation**: This tutorial and inline docstrings

---

*Happy agent building! 🤖*

*"The best agents are the ones you forget are agents"* - FlowGen Philosophy