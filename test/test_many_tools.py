import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from flowgen.llm.llm import LLM
from flowgen.agent.agent import Agent

def get_weather(location: str) -> str:
    return f"Weather in {location}: 22°C, sunny"

def calculate_sum(a: float, b: float) -> float:
    return a + b

def get_time(timezone: str = "UTC") -> str:
    return f"Current time in {timezone}: 14:30"

def search_web(query: str) -> str:
    return f"Search results for '{query}': Found 3 relevant articles"

def translate_text(text: str, target_lang: str = "en") -> str:
    return f"Translated '{text}' to {target_lang}: Hello world"

def test_many_tools():
    """Test UI behavior with many tool calls."""
    print("=== Testing Many Tool Calls ===")
    
    llm = LLM(base_url="http://127.0.0.1:8000")
    agent = Agent(
        llm=llm,
        tools=[get_weather, calculate_sum, get_time, search_web, translate_text],
        stream=True,
        enable_rich_debug=False
    )
    
    # Test query that should trigger multiple tools
    query = "Get weather in Paris, calculate 15+25, check time in PST, search for 'AI news', and translate 'hello' to French"
    
    print(f"Query: {query}")
    print("Simulating tool call responses...")
    
    # Simulate the UI receiving many tool events
    tool_events = [
        {'type': 'tool_start', 'tool_name': 'get_weather', 'tool_args': {'location': 'Paris'}},
        {'type': 'tool_result', 'tool_name': 'get_weather', 'tool_args': {'location': 'Paris'}, 'result': 'Weather in Paris: 22°C, sunny'},
        {'type': 'tool_start', 'tool_name': 'calculate_sum', 'tool_args': {'a': 15, 'b': 25}},
        {'type': 'tool_result', 'tool_name': 'calculate_sum', 'tool_args': {'a': 15, 'b': 25}, 'result': '40'},
        {'type': 'tool_start', 'tool_name': 'get_time', 'tool_args': {'timezone': 'PST'}},
        {'type': 'tool_result', 'tool_name': 'get_time', 'tool_args': {'timezone': 'PST'}, 'result': 'Current time in PST: 14:30'},
        {'type': 'tool_start', 'tool_name': 'search_web', 'tool_args': {'query': 'AI news'}},
        {'type': 'tool_result', 'tool_name': 'search_web', 'tool_args': {'query': 'AI news'}, 'result': 'Search results for AI news: Found 3 relevant articles'},
        {'type': 'tool_start', 'tool_name': 'translate_text', 'tool_args': {'text': 'hello', 'target_lang': 'French'}},
        {'type': 'tool_result', 'tool_name': 'translate_text', 'tool_args': {'text': 'hello', 'target_lang': 'French'}, 'result': 'Translated hello to French: Bonjour'},
    ]
    
    # Simulate the UI processing
    result = []
    tool_calls_completed = []
    tool_call_count = 0
    
    for event in tool_events:
        event_type = event.get('type')
        
        if event_type == 'tool_result':
            tool_name = event.get('tool_name')
            tool_args = event.get('tool_args', {})
            tool_result = event.get('result', '')
            tool_call_count += 1
            
            tool_calls_completed.append({
                'name': tool_name,
                'args': tool_args,
                'result': tool_result
            })
            
            print(f"\nAfter tool {tool_call_count} ({tool_name}):")
            
            if tool_call_count <= 2:
                print(f"  Individual display: 🔧 {tool_name}")
            else:
                print(f"  Grouped display: 🔧 {len(tool_calls_completed)-1} Tools + 🔧 {tool_name}")
    
    print(f"\nFinal tool count: {tool_call_count}")
    print("UI would show:")
    if tool_call_count <= 2:
        for tool in tool_calls_completed:
            print(f"  🔧 {tool['name']} (expanded)")
    else:
        print(f"  🔧 {len(tool_calls_completed)-1} Tools (collapsed summary)")
        print(f"  🔧 {tool_calls_completed[-1]['name']} (expanded)")

if __name__ == "__main__":
    test_many_tools()