#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flowgen.llm.llm import LLM

def test_llm_basic():
    """Test basic LLM functionality."""
    print("=== Testing LLM Basic Functionality ===")
    
    llm = LLM(base_url="http://localhost:8000")
    
    try:
        # Test simple text generation
        result = llm("What is 2+2?", max_tokens=500)
        print(f"✅ Basic chat successful")
        print(f"Response: {result}")
        
        # Check response format
        if 'content' in result:
            print(f"✅ Response format correct")
        else:
            print(f"❌ Missing 'content' in response: {result}")
            
    except Exception as e:
        print(f"❌ Basic chat failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    test_llm_basic()