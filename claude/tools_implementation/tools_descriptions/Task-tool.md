# Task Tool - Comprehensive Documentation

## Overview
The Task tool launches specialized agents to handle complex, multi-step tasks autonomously. It's designed for delegation of work that requires multiple rounds of tool usage, research, or complex problem-solving.

## Function Signature
```json
{
  "name": "Task",
  "parameters": {
    "description": "string (required, 3-5 words)",
    "prompt": "string (required)",
    "subagent_type": "string (required)"
  }
}
```

## Parameters

### description (required)
- **Type**: string
- **Constraints**: 3-5 words only
- **Purpose**: Brief summary of the task for tracking/logging
- **Validation**: Must be concise descriptive phrase
- **Examples**:
  - "Search for functions"
  - "Analyze code structure"
  - "Fix build errors"

### prompt (required)
- **Type**: string
- **Purpose**: Detailed instructions for the agent to execute autonomously
- **Constraints**: Must be self-contained and complete
- **Best Practices**:
  - Include specific deliverables expected
  - Specify format of return information
  - Provide context and background
  - Be explicit about what to search for or analyze

### subagent_type (required)
- **Type**: string
- **Available Values**: 
  - `"general-purpose"` - Has access to all tools (*)
- **Capabilities**: Research, code search, file operations, web fetching, multi-step analysis

## Agent Capabilities

### General-Purpose Agent
- **Tools Available**: All 17 tools in the system
- **Use Cases**:
  - Complex codebase research
  - Multi-file analysis
  - Code pattern searching
  - Implementation planning
  - Bug investigation
- **Limitations**: Single stateless interaction, no follow-up communication

## When to Use Task Tool

### Recommended Scenarios
1. **Open-ended searches** requiring multiple search attempts
2. **Complex research** spanning multiple files/directories
3. **Multi-step analysis** that might require tool chaining
4. **Slash command execution** (custom commands starting with /)
5. **Uncertain search scope** where you're not confident about finding matches quickly

### When NOT to Use
1. **Specific file reads** - Use Read tool directly
2. **Known class/function searches** - Use Glob tool for faster results
3. **Limited scope searches** (2-3 specific files) - Use Read tool
4. **Simple single-step tasks**

## Behavior and Constraints

### Execution Model
- **Stateless**: Each invocation is independent
- **Single Response**: Agent returns one final message
- **Autonomous**: No mid-task communication possible
- **Parallel Execution**: Multiple agents can run concurrently

### Performance Optimization
- **Concurrent Agents**: Launch multiple agents in single message when possible
- **Context Efficiency**: Reduces main conversation context usage
- **Specialized Processing**: Agents optimize for their specific task type

### Return Value Processing
- Agent results are NOT visible to user
- You must summarize/relay important findings to user
- Trust agent outputs - they are generally reliable
- Extract actionable information for next steps

## Usage Patterns

### Basic Research
```json
{
  "description": "Find error handlers",
  "prompt": "Search the codebase for all error handling patterns. Look for try-catch blocks, error callbacks, and custom error classes. Return a summary of the main error handling approaches used.",
  "subagent_type": "general-purpose"
}
```

### Slash Command Execution
```json
{
  "description": "Execute custom command",
  "prompt": "/check-file path/to/file.py",
  "subagent_type": "general-purpose"
}
```

### Complex Analysis
```json
{
  "description": "Analyze API structure",
  "prompt": "Examine the API layer of this application. Find all route definitions, middleware usage, and request/response patterns. Create a comprehensive map of the API structure including endpoints, methods, and data flow.",
  "subagent_type": "general-purpose"
}
```

## Advanced Features

### Parallel Agent Execution
```javascript
// Launch multiple agents simultaneously
[
  Task({description: "Find database code", prompt: "...", subagent_type: "general-purpose"}),
  Task({description: "Find API routes", prompt: "...", subagent_type: "general-purpose"}),
  Task({description: "Find test files", prompt: "...", subagent_type: "general-purpose"})
]
```

### Research vs Implementation Guidance
- **Research Tasks**: Explicitly state "research and analyze, do not write code"
- **Implementation Tasks**: Clearly specify "implement the following feature"
- **Mixed Tasks**: Break into research phase then implementation phase

## Error Handling

### Common Failure Modes
1. **Vague prompts** - Agent may not find relevant information
2. **Tool limitations** - Some operations may not be possible
3. **Context overflow** - Very large codebases may hit limits
4. **Permission issues** - File access restrictions

### Best Practices for Reliability
1. **Specific prompts** with clear success criteria
2. **Fallback planning** - have alternative approaches ready
3. **Incremental requests** - break large tasks into smaller parts
4. **Context awareness** - understand what agent can/cannot access

## Integration Patterns

### With Other Tools
- **Follow-up actions**: Use agent results to inform direct tool usage
- **Validation**: Use Read tool to verify agent findings
- **Implementation**: Agent research → direct tool implementation

### Workflow Examples
1. **Bug Investigation**:
   - Task agent: Research error patterns
   - Review agent findings
   - Read specific files identified
   - Edit files with fixes

2. **Feature Implementation**:
   - Task agent: Analyze existing similar features
   - Plan implementation based on findings
   - Use direct tools for implementation
   - Task agent: Validate implementation

## Performance Considerations

### Optimization Strategies
- **Batch research**: Combine related research into single agent call
- **Scope limiting**: Provide specific directories or file patterns when possible
- **Context management**: Use agents to reduce main conversation context
- **Tool selection**: Use Task for complex searches, direct tools for known operations

### Cost Implications
- Agents consume additional compute resources
- Consider direct tool usage for simple operations
- Balance thoroughness vs efficiency based on task complexity

## Security and Safety

### Limitations
- Agents inherit same security restrictions as main session
- No additional permissions or capabilities
- Cannot access external systems beyond available tools
- Subject to same content filtering and safety measures

### Best Practices
- Avoid sensitive information in prompts
- Review agent outputs before acting on recommendations
- Maintain principle of least privilege in task scoping