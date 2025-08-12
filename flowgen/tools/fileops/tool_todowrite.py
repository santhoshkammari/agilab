import json
from typing import TypedDict,Literal,List

class Todo(TypedDict):
    id:str
    content:str
    priority:Literal['high','medium','low']
    status:Literal['pending','in_progress','completed','cancelled']

def todo_write(todos:List[Todo]):
    """
    Create and manage structured task lists for coding sessions.
    
    Args:
        todos (list): List of todo objects, each containing:
            - content (str): Task description (required, min 1 character)
            - status (str): Task status - 'pending', 'in_progress', or 'completed' (required)
            - priority (str): Task priority - 'high', 'medium', or 'low' (required)
            - id (str): Unique identifier for the task (required)
    
    Returns:
        dict: Success response with task summary
        
    Raises:
        ValueError: If todos list is invalid or contains invalid todo objects
        TypeError: If todos is not a list
    """
    if not isinstance(todos, list):
        try:
            todos = json.loads(todos)
        except:
            # raise TypeError("todos must be a list")
            return "I need the todos parameter to be a list of todo objects, but I received a different data type."

    if len(todos) == 0:
        # raise ValueError("todos list must contain at least 1 todo object")
        return "I need at least one todo item in the list to create a task list."

    # Validate each todo object
    valid_statuses = {'pending', 'in_progress', 'completed'}
    valid_priorities = {'high', 'medium', 'low'}
    in_progress_count = 0
    todo_ids = set()
    
    for i, todo in enumerate(todos):
        if not isinstance(todo, dict):
            # raise ValueError(f"Todo at index {i} must be a dictionary")
            return f"I found an invalid todo item at position {i}. Each todo needs to be a dictionary with the required fields."

        # Check required fields
        required_fields = ['content', 'status', 'priority', 'id']
        for field in required_fields:
            if field not in todo:
                # raise ValueError(f"Todo at index {i} missing required field: {field}")
                return f"The todo item at position {i} is missing the required '{field}' field. Each todo needs content, status, priority, and id."

        # Validate content
        content = todo['content']
        if not isinstance(content, str) or len(content.strip()) == 0:
            # raise ValueError(f"Todo at index {i}: content must be a non-empty string")
            return f"The todo item at position {i} needs meaningful content. Please provide a non-empty description for the task."

        # Validate status
        status = todo['status']
        if status not in valid_statuses:
#             raise ValueError(f"Todo at index {i}: status must be one of {valid_statuses}")
            return f"The todo item at position {i} has an invalid status. Please use one of: {', '.join(valid_statuses)}."

        if status == 'in_progress':
            in_progress_count += 1
        
        # Validate priority
        priority = todo['priority']
        if priority not in valid_priorities:
#             raise ValueError(f"Todo at index {i}: priority must be one of {valid_priorities}")
            return f"The todo item at position {i} has an invalid priority. Please use one of: {', '.join(valid_priorities)}."

        # Validate id
        todo_id = todo['id']
        if not isinstance(todo_id, str) or len(todo_id.strip()) == 0:
#             raise ValueError(f"Todo at index {i}: id must be a non-empty string")
            return f"The todo item at position {i} needs a unique identifier. Please provide a non-empty id string."

        if todo_id in todo_ids:
#             raise ValueError(f"Todo at index {i}: duplicate id '{todo_id}'")
            return f"I found a duplicate id '{todo_id}' at position {i}. Each todo item needs a unique identifier."
        todo_ids.add(todo_id)
    
    # Validate only one in_progress task
    if in_progress_count > 1:
#         raise ValueError(f"Only one task can be 'in_progress' at a time, found {in_progress_count}")
        return f"I can only track one task as 'in_progress' at a time, but I found {in_progress_count} tasks marked as in progress. Please update the statuses so only one task is actively being worked on."

    # Count tasks by status and priority
    status_counts = {'pending': 0, 'in_progress': 0, 'completed': 0}
    priority_counts = {'high': 0, 'medium': 0, 'low': 0}
    
    for todo in todos:
        status_counts[todo['status']] += 1
        priority_counts[todo['priority']] += 1
    
    # Create success response
    response = {
        'success': True,
        'message': f"Successfully managed {len(todos)} todo{'s' if len(todos) != 1 else ''}",
        'summary': {
            'total_tasks': len(todos),
            'status_breakdown': status_counts,
            'priority_breakdown': priority_counts,
            'current_task': None
        },
        'todos': todos
    }
    
    # Add current task info if there's one in progress
    for todo in todos:
        if todo['status'] == 'in_progress':
            response['summary']['current_task'] = {
                'id': todo['id'],
                'content': todo['content'],
                'priority': todo['priority']
            }
            break
    
    return response

