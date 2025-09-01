#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent
import json
import time

def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 22°C, sunny, humidity 65%"

def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers"""
    return a + b

def test_basic_chat():
    """Test 1: Basic chat conversation using Agent"""
    print("\n=== Test 1: Basic Agent Chat ===")
    
    llm = LLM()
    agent = Agent(llm=llm, enable_rich_debug=True)
    
    response = agent("Hello! What's the capital of France?", max_tokens=50, temperature=0.1)
    
    print(f"Agent Response: {json.dumps(response, indent=2)}")
    print(f"Content: {response.get('content', '')}")
    print(f"Iterations: {response.get('iterations', 0)}")
    
    return response

def test_multiturn_conversation():
    """Test 2: Multi-turn conversation with message building using Agent"""
    print("\n=== Test 2: Multi-turn Agent Conversation ===")
    
    llm = LLM()
    agent = Agent(llm=llm, enable_rich_debug=True)
    
    # First interaction
    response1 = agent("I'm planning a trip to Japan.", max_tokens=100, temperature=0.3)
    print(f"Agent 1: {response1.get('content', '')}")
    
    # Get conversation so far
    conversation = agent.get_conversation()
    print(f"Conversation length after first exchange: {len(conversation)}")
    
    # Continue conversation by adding to history
    agent.add_history(conversation)
    
    # Second interaction
    response2 = agent("What's the best time to visit for cherry blossoms?", max_tokens=100, temperature=0.3)
    print(f"Agent 2: {response2.get('content', '')}")
    
    # Continue building
    conversation = agent.get_conversation()
    agent.add_history(conversation)
    
    # Third interaction
    response3 = agent("Thank you! Any food recommendations?", max_tokens=100, temperature=0.3)
    print(f"Agent 3: {response3.get('content', '')}")
    
    final_conversation = agent.get_conversation()
    print(f"Final conversation length: {len(final_conversation)} messages")
    
    return final_conversation

def test_structured_json():
    """Test 3: Structured JSON response format using Agent"""
    print("\n=== Test 3: Agent Structured JSON Response ===")
    
    llm = LLM()
    agent = Agent(llm=llm, enable_rich_debug=True)
    
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "number"},
            "famous_for": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["city", "country"]
    }
    
    response = agent("Tell me about Tokyo in the specified JSON format", 
                    format=schema, max_tokens=200, temperature=0.1)
    
    print(f"Agent JSON Response: {json.dumps(response, indent=2)}")
    
    content = response.get('content', '')
    if content:
        try:
            parsed_json = json.loads(content)
            print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw content: {content}")
    
    return response

def test_tool_calling():
    """Test 4: Tool calling with get_weather and calculate_sum using Agent"""
    print("\n=== Test 4: Agent Tool Calling ===")
    
    llm = LLM()
    agent = Agent(llm=llm, tools=[get_weather, calculate_sum], enable_rich_debug=True)
    
    response = agent("What's the weather in Tokyo and what's 15 + 27?", 
                    max_tokens=300, temperature=0.1)
    
    print(f"Agent Tool Call Response: {json.dumps(response, indent=2)}")
    
    # Check conversation for tool execution
    conversation = agent.get_conversation()
    print(f"\nConversation with {len(conversation)} messages:")
    
    for i, msg in enumerate(conversation):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            print(f"{i+1}. User: {content}")
        elif role == 'assistant':
            if 'tool_calls' in msg:
                tool_names = [tc['function']['name'] for tc in msg['tool_calls']]
                print(f"{i+1}. Assistant: [Called tools: {', '.join(tool_names)}]")
            else:
                print(f"{i+1}. Assistant: {content}")
        elif role == 'tool':
            tool_name = msg.get('name', 'unknown')
            print(f"{i+1}. Tool ({tool_name}): {content}")
    
    return response

def test_streaming():
    """Test 5: Streaming agent"""
    print("\n=== Test 5: Streaming Agent ===")
    
    llm = LLM()
    agent = Agent(llm=llm, tools=[get_weather], stream=True, enable_rich_debug=True)
    
    print("Starting streaming...")
    for event in agent("Get weather for London and tell me about it", stream=True, max_tokens=150):
        event_type = event.get('type', 'unknown')
        
        if event_type == 'token':
            print(f"Token: '{event.get('content', '')}'", end='', flush=True)
        elif event_type == 'tool_start':
            print(f"\n🔧 Starting tool: {event.get('tool_name', 'unknown')}")
        elif event_type == 'tool_result':
            print(f"🔧 Tool result: {event.get('result', '')}")
        elif event_type == 'final':
            print(f"\n🏁 Final: {event.get('content', '')}")
            print(f"Iterations: {event.get('iterations', 0)}")
            break
    
    return agent.get_conversation()

def main():
    print("Testing Flowgen LLM and Agent classes")
    
    try:
        # Test LLM directly
        test_basic_chat()
        time.sleep(1)
        
        test_multiturn_conversation()
        time.sleep(1)
        
        test_structured_json()
        time.sleep(1)
        
        test_tool_calling()
        time.sleep(1)
        
        test_streaming()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== All Testing Complete ===")

if __name__ == "__main__":
    main()