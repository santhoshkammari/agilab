# Agilab

Agilab is an advanced AI framework designed for creating autonomous agents with sophisticated tool usage, conversation management, and behavioral dataset generation capabilities. Built on a foundation of agentic architectures, it provides a comprehensive platform for developing, training, and deploying AI agents that can interact with complex environments and tools.

## Introduction

Agilab provides a flexible and extensible framework for building autonomous AI agents. The system is built around the concept of agentic workflows, where agents can intelligently use tools, maintain conversation history, and execute complex tasks with minimal human intervention. The framework supports multiple LLM backends and provides streaming, batching, and persistence capabilities for production-grade applications.

The core components include:
- **Agent Framework**: A flexible system for creating autonomous agents with tool usage capabilities
- **LLM Integration**: Support for multiple LLM providers through a unified interface
- **Tool Management**: Dynamic tool loading and execution system
- **Conversation Management**: History tracking and serialization
- **Streaming Support**: Real-time interaction with agents

## Roadmap

### Current Capabilities
- Agentic tool usage with automatic tool calling and response processing
- Support for multiple LLM providers (vLLM, Gemini, Ollama, etc.)
- Streaming and batch processing capabilities
- Agent chaining and composition
- Conversation serialization and persistence
- Built-in tool ecosystem for common operations

### Planned Enhancements

#### Behavioral Induced Dataset Generation
A key focus for Agilab's evolution is the development of tools for generating conversational training datasets through behavioral induction. This includes:

- **Tool Creation Agent**: A specialized agent that can generate and validate tools in multiple formats (YAML and JSONSchema) with built-in schema validation to ensure proper tool definitions and consistent behavior.

- **Tool Response Generation Agent**: An agent that mimics tool results based on query inputs, enabling the simulation of various desired outcomes and scenarios. This includes:
  - Complex outcomes mimicking real tool behavior
  - Failure scenarios to train robust error handling
  - Variations in response patterns to increase dataset diversity
  - Different response types for the same query to simulate real-world variability

- **Conversational Dataset Generation**: By combining the Tool Creation Agent and Tool Response Generation Agent, Agilab will be able to produce high-quality conversations that can be used for training next-generation AI agents. These datasets will include:
  - Diverse tool usage patterns
  - Complex multi-step reasoning scenarios
  - Error handling and recovery patterns
  - Various user interaction styles

#### Future Extensions
- Enhanced multi-agent collaboration systems
- Advanced memory and context management
- Improved serialization formats for ML training pipelines
- Integration with popular ML frameworks for training dataset generation
- Fine-tuning utilities for custom agent behaviors
- Web-based UI for agent management and interaction

## Quick Start

```python
from src.agi import Agent, LLM

# Create an LLM instance
llm = LLM(base_url='http://localhost:8000')

# Create an agent with tools
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    # Your implementation here
    pass

agent = Agent(llm, tools=[calculate])

# Use the agent
result = agent("Calculate 15 * 23")
print(result['content'])
```

## Architecture

The Agilab framework consists of several interconnected components:

- **Agent**: The core class that provides agentic behavior with tool usage, conversation management, and streaming capabilities
- **LLM**: Abstract base class and implementations for connecting to various LLM providers
- **Tools**: Various utility modules providing common functionality
- **Template**: System for managing conversation templates
- **API**: Interface components for integrating with different services

## Contributing

We welcome contributions to the Agilab project. Please see our contributing guidelines for more information on how to get involved.

## License

This project is licensed under the MIT License.