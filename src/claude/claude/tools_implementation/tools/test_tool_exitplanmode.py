from tool_exitplanmode import exit_plan_mode

def test_exit_plan_mode():
    """Test the exit_plan_mode function with various scenarios."""
    
    # Test 1: Basic plan formatting
    plan = """## Authentication Implementation

### Steps
1. Set up database models
2. Create API endpoints
3. Implement frontend components

### Testing
- Unit tests for auth functions
- Integration tests for endpoints"""
    
    result = exit_plan_mode(plan)
    assert "## Implementation Plan Ready for Approval" in result
    assert "## Authentication Implementation" in result
    assert "Ready to proceed?" in result
    print("✓ Basic plan formatting test passed")
    
    # Test 2: Plan with markdown formatting
    plan_with_markdown = """# Feature Implementation Plan

## Overview
Implement **user authentication** with the following features:
- Login/logout functionality
- Password reset
- JWT token management

## Technical Approach
- Use `bcrypt` for password hashing
- Implement rate limiting
- Add email validation

### Code Example
```javascript
const hashPassword = await bcrypt.hash(password, 10);
```

## Success Criteria
- [ ] Users can register
- [ ] Users can login
- [ ] Password reset works"""
    
    result = exit_plan_mode(plan_with_markdown)
    assert "# Feature Implementation Plan" in result
    assert "```javascript" in result
    assert "- [ ] Users can register" in result
    assert "Ready to proceed?" in result
    print("✓ Markdown formatting preservation test passed")
    
    # Test 3: Simple one-line plan
    simple_plan = "Fix the memory leak in the data processing module"
    result = exit_plan_mode(simple_plan)
    assert "Fix the memory leak" in result
    assert "Implementation Plan Ready for Approval" in result
    print("✓ Simple plan test passed")
    
    # Test 4: Plan with extra whitespace
    plan_with_whitespace = """
    
    ## Bug Fix Plan
    
    Fix the authentication issue by updating the token validation logic.
    
    """
    result = exit_plan_mode(plan_with_whitespace)
    assert result.count("## Bug Fix Plan") == 1  # Should appear only once, whitespace trimmed
    assert not result.startswith("\n")  # Should not start with newlines
    print("✓ Whitespace handling test passed")
    
    # Test 5: Empty plan validation
    try:
        exit_plan_mode("")
        assert False, "Should have raised ValueError for empty plan"
    except ValueError as e:
        assert "cannot be empty" in str(e)
        print("✓ Empty plan validation test passed")
    
    # Test 6: Whitespace-only plan validation  
    try:
        exit_plan_mode("   \n\t   ")
        assert False, "Should have raised ValueError for whitespace-only plan"
    except ValueError as e:
        assert "cannot be empty" in str(e)
        print("✓ Whitespace-only plan validation test passed")
    
    # Test 7: Non-string plan validation
    try:
        exit_plan_mode(123)
        assert False, "Should have raised TypeError for non-string plan"
    except TypeError as e:
        assert "must be a string" in str(e)
        print("✓ Non-string plan validation test passed")
    
    # Test 8: None plan validation
    try:
        exit_plan_mode(None)
        assert False, "Should have raised TypeError for None plan"
    except TypeError as e:
        assert "must be a string" in str(e)
        print("✓ None plan validation test passed")
    
    # Test 9: Long plan handling
    long_plan = """## Comprehensive Feature Implementation Plan

### Phase 1: Database Setup
""" + "".join(f"Step {i}: Description of step {i}\n" for i in range(1, 101))
    long_plan = long_plan.rstrip()  # Remove trailing newline
    
    result = exit_plan_mode(long_plan)
    assert "Phase 1: Database Setup" in result
    assert "Step 50:" in result  # Verify long content is preserved
    assert "Ready to proceed?" in result
    print("✓ Long plan handling test passed")
    
    # Test 10: Plan with special characters
    special_plan = """## Plan with Special Characters

### Steps
1. Handle UTF-8 characters: café, résumé, naïve
2. Process symbols: @#$%^&*()
3. Handle quotes: "double" and 'single'
4. Process backslashes: C:\\path\\to\\file

### Notes
- Use proper escaping for SQL queries
- Handle Unicode normalization"""
    
    result = exit_plan_mode(special_plan)
    assert "café, résumé, naïve" in result
    assert "@#$%^&*()" in result
    assert '"double" and \'single\'' in result
    assert "C:\\path\\to\\file" in result
    print("✓ Special characters test passed")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_exit_plan_mode()