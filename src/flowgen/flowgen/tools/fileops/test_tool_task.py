from tool_task import task

def test_task():
    """Test the task function with various scenarios."""
    
    # Test 1: Basic general-purpose task (search)
    description = "Find error handlers"
    prompt = "Search the codebase for all error handling patterns. Look for try-catch blocks, error callbacks, and custom error classes."
    subagent_type = "general-purpose"
    
    result = task(description, prompt, subagent_type)
    assert "Agent completed search task" in result
    assert description in result
    assert "Completed" in result
    print("✓ Basic search task test passed")
    
    # Test 2: Analysis task
    description = "Analyze API structure"
    prompt = "Examine the API layer of this application. Find all route definitions, middleware usage, and request/response patterns."
    subagent_type = "general-purpose"
    
    result = task(description, prompt, subagent_type)
    assert "Agent completed analysis task" in result
    assert description in result
    print("✓ Analysis task test passed")
    
    # Test 3: Slash command task
    description = "Execute custom command"
    prompt = "/check-file path/to/file.py"
    subagent_type = "general-purpose"
    
    result = task(description, prompt, subagent_type)
    assert "Agent executed command task" in result
    assert description in result
    print("✓ Slash command task test passed")
    
    # Test 4: General task
    description = "Fix build errors"
    prompt = "Handle any build errors in the project. Check configuration files and dependencies."
    subagent_type = "general-purpose"
    
    result = task(description, prompt, subagent_type)
    assert "Agent completed general task" in result
    assert description in result
    print("✓ General task test passed")
    
    # Test 5: Description validation - too few words (2 words)
    try:
        task("Fix errors", "Some prompt", "general-purpose")
        assert False, "Should have raised ValueError for too few words"
    except ValueError as e:
        assert "3-5 words only" in str(e)
        print("✓ Too few words validation test passed")
    
    # Test 6: Description validation - too many words (6 words)
    try:
        task("Fix all the build errors now", "Some prompt", "general-purpose")
        assert False, "Should have raised ValueError for too many words"
    except ValueError as e:
        assert "3-5 words only" in str(e)
        print("✓ Too many words validation test passed")
    
    # Test 7: Description validation - exactly 3 words (minimum)
    result = task("Fix build errors", "Some prompt", "general-purpose")
    assert "Agent completed" in result
    print("✓ Minimum word count (3) test passed")
    
    # Test 8: Description validation - exactly 5 words (maximum)
    result = task("Analyze code structure patterns completely", "Some prompt", "general-purpose")
    assert "Agent completed" in result
    print("✓ Maximum word count (5) test passed")
    
    # Test 9: Invalid subagent_type
    try:
        task("Fix errors now", "Some prompt", "invalid-type")
        assert False, "Should have raised ValueError for invalid subagent_type"
    except ValueError as e:
        assert "subagent_type must be one of" in str(e)
        print("✓ Invalid subagent_type validation test passed")
    
    # Test 10: Empty prompt
    try:
        task("Fix errors now", "", "general-purpose")
        assert False, "Should have raised ValueError for empty prompt"
    except ValueError as e:
        assert "prompt cannot be empty" in str(e)
        print("✓ Empty prompt validation test passed")
    
    # Test 11: Whitespace-only prompt
    try:
        task("Fix errors now", "   ", "general-purpose")
        assert False, "Should have raised ValueError for whitespace-only prompt"
    except ValueError as e:
        assert "prompt cannot be empty" in str(e)
        print("✓ Whitespace-only prompt validation test passed")
    
    # Test 12: Type validation - non-string description
    try:
        task(123, "Some prompt", "general-purpose")
        assert False, "Should have raised TypeError for non-string description"
    except TypeError as e:
        assert "description must be a string" in str(e)
        print("✓ Non-string description validation test passed")
    
    # Test 13: Type validation - non-string prompt
    try:
        task("Fix errors now", 123, "general-purpose")
        assert False, "Should have raised TypeError for non-string prompt"
    except TypeError as e:
        assert "prompt must be a string" in str(e)
        print("✓ Non-string prompt validation test passed")
    
    # Test 14: Type validation - non-string subagent_type
    try:
        task("Fix errors now", "Some prompt", 123)
        assert False, "Should have raised TypeError for non-string subagent_type"
    except TypeError as e:
        assert "subagent_type must be a string" in str(e)
        print("✓ Non-string subagent_type validation test passed")
    
    # Test 15: Response structure validation
    result = task("Search for functions", "Find all function definitions", "general-purpose")
    
    # Check that response contains expected structure
    assert "Task Agent Response:" in result
    assert "Description:" in result
    assert "Type:" in result
    assert "Capabilities:" in result
    assert "Task Status:" in result
    assert "Agent Tools Used:" in result
    assert "Execution Mode:" in result
    print("✓ Response structure validation test passed")
    
    # Test 16: Capabilities included in response
    result = task("Analyze code structure", "Examine the code", "general-purpose")
    assert "Research" in result
    assert "code search" in result
    assert "file operations" in result
    assert "web fetching" in result
    assert "multi-step analysis" in result
    print("✓ Capabilities inclusion test passed")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_task()