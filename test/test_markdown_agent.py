#!/usr/bin/env python3

import sys
import os
import json
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent
from flowgen.tools.markdown_content import tool_functions as markdown_tools


def simple_calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"


class MockLLM(LLM):
    """Mock LLM for testing without a real server."""
    
    def __init__(self):
        super().__init__(base_url="http://localhost:8000")
    
    def __call__(self, messages):
        # Check if there are tool calls in the messages
        for msg in reversed(messages):
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                # Return a response that references the tool results
                return {
                    'content': "I've analyzed the markdown content and provided the results above.",
                    'tools': []
                }
        
        # First call - return a response with tool calls
        return {
            'content': '',
            'tools': [
                {
                    'id': 'call_1',
                    'name': 'markdown_analyzer_get_headers',
                    'arguments': {
                        'content': messages[-1]['content']
                    }
                }
            ]
        }


def test_markdown_agent():
    """Test Agent with markdown tools using a mock LLM."""
    print("=== Testing Agent with Markdown Tools (Mock LLM) ===")
    
    # Combine calculator tool with markdown analysis tools
    all_tools = [simple_calculator] + list(markdown_tools.values())
    
    llm = MockLLM()
    agent = Agent(llm=llm, tools=all_tools, stream=False, enable_rich_debug=False)
    
    try:
        # Sample markdown content for testing
        sample_md = """# Test Document

## Introduction
This is a sample markdown document for testing.

### Subsection
Some content here.

```python
print("Hello World")
```

- Item 1
- Item 2
"""
        
        prompt = f"Analyze this markdown content and show me the headers:\n\n{sample_md}"
        
        print("Sending prompt to agent...")
        result = agent(prompt)
        
        print(f"✅ Agent completed successfully")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Response: {result.get('content', '')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_markdown_agent()