#!/usr/bin/env python3
"""
Test the exact CLI scenario that's failing
"""

from claude.llm import OAI
from claude.tools import tools
from claude.config import config

def test_cli_scenario():
    """Test exact CLI scenario with Google provider"""
    print("=== Testing CLI Scenario ===")
    
    # Set up Google provider like in CLI
    config.set_provider("google")
    
    try:
        # Initialize LLM like in input.py
        provider_config = config.get_current_config()
        llm = OAI(
            provider="google", 
            api_key=provider_config["api_key"],
            model=provider_config["model"]
        )
        
        # Set up conversation like in CLI
        conversation_history = [
            {
                "role": "system", 
                "content": "You are Claude Code, a helpful AI assistant for programming tasks."
            },
            {
                "role": "user",
                "content": "search for 'who won ipl 2023'"
            }
        ]
        
        print("Making request with tools like CLI does...")
        print(f"Tools count: {len(tools)}")
        print(f"Model: {config.model}")
        
        # This is the exact call from input.py line 979
        response = llm.run(
            messages=conversation_history, 
            model=config.model, 
            max_tokens=config.num_ctx, 
            tools=tools
        )
        
        print("✅ CLI scenario successful!")
        
        if response.choices[0].message.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
        else:
            print("No tool calls - response:", response.choices[0].message.content)
        
        return True
        
    except Exception as e:
        print(f"❌ CLI scenario failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_values():
    """Test config values"""
    print("\n=== Testing Config Values ===")
    
    try:
        config.set_provider("google")
        
        print(f"Provider: {config.provider}")
        print(f"Config: {config.get_current_config()}")
        print(f"Model: {config.model}")
        print(f"Num ctx: {config.num_ctx}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    test_config_values()
    test_cli_scenario()