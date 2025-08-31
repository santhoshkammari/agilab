from tool_todowrite import todo_write

def test_todo_write():
    """Test the todo_write function with various scenarios."""
    
    # Test 1: Valid single todo
    todos = [
        {
            "content": "Implement user authentication",
            "status": "pending",
            "priority": "high",
            "id": "auth-implementation"
        }
    ]
    
    result = todo_write(todos)
    assert result['success'] == True
    assert result['summary']['total_tasks'] == 1
    assert result['summary']['status_breakdown']['pending'] == 1
    assert result['summary']['priority_breakdown']['high'] == 1
    assert result['summary']['current_task'] is None
    print("✓ Single todo test passed")
    
    # Test 2: Multiple todos with one in progress
    todos = [
        {
            "content": "Set up project structure",
            "status": "completed",
            "priority": "high",
            "id": "project-setup"
        },
        {
            "content": "Implement authentication system",
            "status": "in_progress",
            "priority": "high",
            "id": "auth-system"
        },
        {
            "content": "Add unit tests",
            "status": "pending",
            "priority": "medium",
            "id": "unit-tests"
        },
        {
            "content": "Deploy to staging",
            "status": "pending",
            "priority": "low",
            "id": "staging-deploy"
        }
    ]
    
    result = todo_write(todos)
    assert result['success'] == True
    assert result['summary']['total_tasks'] == 4
    assert result['summary']['status_breakdown']['completed'] == 1
    assert result['summary']['status_breakdown']['in_progress'] == 1
    assert result['summary']['status_breakdown']['pending'] == 2
    assert result['summary']['priority_breakdown']['high'] == 2
    assert result['summary']['priority_breakdown']['medium'] == 1
    assert result['summary']['priority_breakdown']['low'] == 1
    assert result['summary']['current_task']['id'] == "auth-system"
    print("✓ Multiple todos test passed")
    
    # Test 3: Empty todos list
    try:
        todo_write([])
        assert False, "Should have raised ValueError for empty list"
    except ValueError as e:
        assert "at least 1 todo" in str(e)
        print("✓ Empty list validation test passed")
    
    # Test 4: Invalid input type
    try:
        todo_write("not a list")
        assert False, "Should have raised TypeError for non-list input"
    except TypeError as e:
        assert "must be a list" in str(e)
        print("✓ Invalid input type test passed")
    
    # Test 5: Missing required field
    todos = [
        {
            "content": "Test task",
            "status": "pending",
            "priority": "medium"
            # Missing 'id' field
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for missing field"
    except ValueError as e:
        assert "missing required field: id" in str(e)
        print("✓ Missing field validation test passed")
    
    # Test 6: Invalid status
    todos = [
        {
            "content": "Test task",
            "status": "invalid_status",
            "priority": "medium",
            "id": "test-id"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for invalid status"
    except ValueError as e:
        assert "status must be one of" in str(e)
        print("✓ Invalid status validation test passed")
    
    # Test 7: Invalid priority
    todos = [
        {
            "content": "Test task",
            "status": "pending",
            "priority": "urgent",
            "id": "test-id"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for invalid priority"
    except ValueError as e:
        assert "priority must be one of" in str(e)
        print("✓ Invalid priority validation test passed")
    
    # Test 8: Empty content
    todos = [
        {
            "content": "",
            "status": "pending",
            "priority": "medium",
            "id": "test-id"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for empty content"
    except ValueError as e:
        assert "content must be a non-empty string" in str(e)
        print("✓ Empty content validation test passed")
    
    # Test 9: Empty id
    todos = [
        {
            "content": "Test task",
            "status": "pending",
            "priority": "medium",
            "id": ""
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for empty id"
    except ValueError as e:
        assert "id must be a non-empty string" in str(e)
        print("✓ Empty id validation test passed")
    
    # Test 10: Duplicate ids
    todos = [
        {
            "content": "First task",
            "status": "pending",
            "priority": "medium",
            "id": "duplicate-id"
        },
        {
            "content": "Second task",
            "status": "pending",
            "priority": "low",
            "id": "duplicate-id"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for duplicate ids"
    except ValueError as e:
        assert "duplicate id" in str(e)
        print("✓ Duplicate id validation test passed")
    
    # Test 11: Multiple in_progress tasks
    todos = [
        {
            "content": "First task",
            "status": "in_progress",
            "priority": "high",
            "id": "task-1"
        },
        {
            "content": "Second task",
            "status": "in_progress",
            "priority": "medium",
            "id": "task-2"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for multiple in_progress tasks"
    except ValueError as e:
        assert "Only one task can be 'in_progress'" in str(e)
        print("✓ Multiple in_progress validation test passed")
    
    # Test 12: Non-dictionary todo object
    todos = [
        "not a dictionary"
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for non-dict todo"
    except ValueError as e:
        assert "must be a dictionary" in str(e)
        print("✓ Non-dictionary todo validation test passed")
    
    # Test 13: All completed tasks
    todos = [
        {
            "content": "Completed task 1",
            "status": "completed",
            "priority": "high",
            "id": "completed-1"
        },
        {
            "content": "Completed task 2",
            "status": "completed",
            "priority": "medium",
            "id": "completed-2"
        }
    ]
    
    result = todo_write(todos)
    assert result['success'] == True
    assert result['summary']['status_breakdown']['completed'] == 2
    assert result['summary']['current_task'] is None
    print("✓ All completed tasks test passed")
    
    # Test 14: Whitespace content validation
    todos = [
        {
            "content": "   ",
            "status": "pending",
            "priority": "medium",
            "id": "whitespace-test"
        }
    ]
    
    try:
        todo_write(todos)
        assert False, "Should have raised ValueError for whitespace-only content"
    except ValueError as e:
        assert "content must be a non-empty string" in str(e)
        print("✓ Whitespace content validation test passed")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_todo_write()