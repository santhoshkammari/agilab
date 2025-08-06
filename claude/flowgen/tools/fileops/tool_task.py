def task_agent(description, prompt, subagent_type):
    """
    Launch specialized agents to handle complex, multi-step tasks autonomously.
    
    Args:
        description (str): Brief summary of the task (3-5 words only)
        prompt (str): Detailed instructions for the agent to execute autonomously
        subagent_type (str): Type of agent to launch ("general-purpose")
        
    Returns:
        str: Agent response with task results
        
    Raises:
        ValueError: If description is not 3-5 words or invalid subagent_type
        TypeError: If required parameters are not strings
    """
    # Validate parameter types
    if not isinstance(description, str):
        raise TypeError("description must be a string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(subagent_type, str):
        raise TypeError("subagent_type must be a string")
    
    # Validate description length (3-5 words)
    words = description.strip().split()
    if len(words) < 3 or len(words) > 5:
        raise ValueError("description must be 3-5 words only")
    
    # Validate subagent_type
    valid_subagent_types = ["general-purpose"]
    if subagent_type not in valid_subagent_types:
        raise ValueError(f"subagent_type must be one of: {valid_subagent_types}")
    
    # Validate prompt is not empty
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    
    # Simulate agent execution
    # In a real implementation, this would launch an actual agent
    # For this mock implementation, we'll return a structured response
    
    if subagent_type == "general-purpose":
        capabilities = [
            "Research", "code search", "file operations", 
            "web fetching", "multi-step analysis"
        ]
        
        # Simulate task processing based on common patterns
        # Check for analysis keywords first as they're more specific
        if any(keyword in prompt.lower() for keyword in ["analyze", "examine", "investigate"]):
            task_type = "analysis"
        elif prompt.startswith("/"):
            task_type = "slash_command"
        elif any(keyword in prompt.lower() for keyword in ["search", "find", "locate"]):
            task_type = "search"
        else:
            task_type = "general"
        
        # Mock response based on task type
        responses = {
            "search": f"Agent completed search task: '{description}'. Found relevant code patterns and structures. Results include file locations, function definitions, and usage examples.",
            "analysis": f"Agent completed analysis task: '{description}'. Examined code structure, patterns, and relationships. Detailed findings include architecture overview and recommendations.",
            "slash_command": f"Agent executed command task: '{description}'. Custom slash command processed successfully with appropriate tool chain execution.",
            "general": f"Agent completed general task: '{description}'. Multi-step task execution completed using available tools and research capabilities."
        }
        
        base_response = responses.get(task_type, responses["general"])
        
        return f"""Task Agent Response:
Description: {description}
Type: {subagent_type}
Capabilities: {', '.join(capabilities)}

{base_response}

Task Status: Completed
Agent Tools Used: Multiple (as per task requirements)
Execution Mode: Autonomous, stateless

Note: This is a simulated response. In production, this would invoke an actual agent with full tool access."""

    # This should never be reached due to validation above
    return f"Unknown subagent type: {subagent_type}"