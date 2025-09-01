# TodoWrite Tool - Comprehensive Documentation

## Overview
The TodoWrite tool creates and manages structured task lists for coding sessions. It helps track progress, organize complex tasks, demonstrate thoroughness, and provides users with clear visibility into task progress and overall request completion.

## Function Signature
```json
{
  "name": "TodoWrite",
  "parameters": {
    "todos": "array of todo objects (required, min 1)"
  }
}
```

## Parameters

### todos (required)
- **Type**: array of todo objects
- **Minimum**: 1 todo object
- **Purpose**: Complete list of tasks to track
- **Behavior**: Replaces entire todo list with provided tasks
- **Structure**: Each todo must include content, status, priority, and id

#### Todo Object Structure
```json
{
  "content": "string (required, min 1 character)",
  "status": "enum (required): pending|in_progress|completed",
  "priority": "enum (required): high|medium|low", 
  "id": "string (required): unique identifier"
}
```

## Todo Object Properties

### content (required)
- **Type**: string
- **Minimum**: 1 character
- **Purpose**: Descriptive task description
- **Format**: Clear, actionable task statement
- **Best Practice**: Specific and measurable

### status (required)
- **Type**: enum
- **Values**: `"pending"` | `"in_progress"` | `"completed"`
- **Purpose**: Current state of the task
- **Constraint**: Only ONE task should be `in_progress` at a time
- **Workflow**: pending → in_progress → completed

### priority (required)
- **Type**: enum
- **Values**: `"high"` | `"medium"` | `"low"`
- **Purpose**: Task importance and urgency
- **Usage**: Helps with task prioritization and organization

### id (required)
- **Type**: string
- **Purpose**: Unique identifier for the task
- **Format**: Descriptive, kebab-case recommended
- **Persistence**: Maintains task identity across updates

## When to Use TodoWrite

### Required Scenarios
1. **Complex Multi-step Tasks**: 3+ distinct steps or actions required
2. **Non-trivial Complex Tasks**: Require careful planning or multiple operations
3. **User Explicit Request**: User directly asks for todo list usage
4. **Multiple Tasks**: User provides numbered or comma-separated task lists
5. **New Instructions**: Immediately capture user requirements as todos
6. **Task Start**: Mark as in_progress BEFORE beginning work
7. **Task Completion**: Mark as completed and add follow-up tasks

### When NOT to Use
1. **Single Straightforward Task**: Simple, one-step operations
2. **Trivial Tasks**: Less than 3 trivial steps
3. **Purely Conversational**: Informational or discussion tasks
4. **No Organizational Benefit**: Tasks that don't benefit from tracking

## Task State Management

### State Transitions
```
pending → in_progress → completed
```

### State Rules
1. **Limit in_progress**: Only ONE task in_progress at any time
2. **Real-time Updates**: Update status as work progresses
3. **Immediate Completion**: Mark completed IMMEDIATELY after finishing
4. **No Batch Completion**: Don't batch multiple completions
5. **Current Task First**: Complete current tasks before starting new ones

### State Definitions
- **pending**: Task not yet started, awaiting attention
- **in_progress**: Currently working on this task (limit: 1)
- **completed**: Task finished successfully

## Task Completion Requirements

### Completion Criteria
Tasks should ONLY be marked completed when:
- **Fully Accomplished**: Task completely finished
- **No Errors**: No unresolved errors or failures
- **Tests Pass**: All relevant tests passing
- **Implementation Complete**: No partial implementation

### Keep in_progress When
- **Encountering Errors**: Unresolved errors or blockers
- **Partial Implementation**: Work still in progress
- **Tests Failing**: Tests need to pass before completion
- **Dependencies Missing**: Required files or resources unavailable

### Blocked Task Handling
```json
{
  "content": "Fix login authentication bug",
  "status": "in_progress",  // Keep in progress
  "priority": "high",
  "id": "auth-bug-fix"
},
{
  "content": "Investigate missing dependency causing auth failure",
  "status": "pending",     // New task for blocker
  "priority": "high", 
  "id": "auth-dependency-issue"
}
```

## Usage Examples

### Project Setup Example
```json
{
  "todos": [
    {
      "content": "Initialize new React project with TypeScript",
      "status": "in_progress",
      "priority": "high",
      "id": "project-init"
    },
    {
      "content": "Configure ESLint and Prettier for code formatting",
      "status": "pending",
      "priority": "high",
      "id": "linting-setup"
    },
    {
      "content": "Set up testing framework with Jest and React Testing Library",
      "status": "pending",
      "priority": "medium",
      "id": "testing-setup"
    },
    {
      "content": "Create basic component structure and routing",
      "status": "pending",
      "priority": "high",
      "id": "component-structure"
    }
  ]
}
```

### Bug Fix Example
```json
{
  "todos": [
    {
      "content": "Reproduce the memory leak issue in development environment",
      "status": "completed",
      "priority": "high",
      "id": "reproduce-memory-leak"
    },
    {
      "content": "Profile memory usage to identify leak source",
      "status": "in_progress", 
      "priority": "high",
      "id": "profile-memory"
    },
    {
      "content": "Fix identified memory leak in event listeners",
      "status": "pending",
      "priority": "high",
      "id": "fix-memory-leak"
    },
    {
      "content": "Add unit tests to prevent regression",
      "status": "pending",
      "priority": "medium",
      "id": "memory-leak-tests"
    },
    {
      "content": "Verify fix in production-like environment",
      "status": "pending",
      "priority": "high",
      "id": "verify-fix"
    }
  ]
}
```

### Feature Implementation Example
```json
{
  "todos": [
    {
      "content": "Design user authentication flow and database schema",
      "status": "completed",
      "priority": "high",
      "id": "auth-design"
    },
    {
      "content": "Implement user registration endpoint with validation",
      "status": "completed",
      "priority": "high", 
      "id": "registration-endpoint"
    },
    {
      "content": "Create login endpoint with JWT token generation",
      "status": "in_progress",
      "priority": "high",
      "id": "login-endpoint"
    },
    {
      "content": "Build password reset functionality",
      "status": "pending",
      "priority": "medium",
      "id": "password-reset"
    },
    {
      "content": "Add comprehensive authentication tests",
      "status": "pending",
      "priority": "high",
      "id": "auth-tests"
    }
  ]
}
```

## Task Breakdown Strategies

### Effective Task Granularity
```json
// ✅ Good - Specific, actionable items
{
  "content": "Implement user profile image upload with validation and resizing",
  "status": "pending",
  "priority": "medium",
  "id": "profile-image-upload"
}

// ❌ Poor - Too vague
{
  "content": "Work on user features",
  "status": "pending", 
  "priority": "medium",
  "id": "user-features"
}
```

### Complex Task Decomposition
```json
// Large task broken into manageable pieces
{
  "todos": [
    {
      "content": "Set up database migrations for shopping cart schema",
      "status": "pending",
      "priority": "high",
      "id": "cart-db-schema"
    },
    {
      "content": "Implement add item to cart API endpoint",
      "status": "pending", 
      "priority": "high",
      "id": "add-to-cart-api"
    },
    {
      "content": "Create remove item from cart functionality",
      "status": "pending",
      "priority": "high",
      "id": "remove-from-cart"
    },
    {
      "content": "Build cart quantity update mechanisms",
      "status": "pending",
      "priority": "medium",
      "id": "cart-quantity-update"
    },
    {
      "content": "Add cart persistence across user sessions",
      "status": "pending",
      "priority": "medium",
      "id": "cart-persistence"
    }
  ]
}
```

## Progress Tracking Workflow

### Real-time Updates
```json
// Starting work - mark in_progress
{
  "content": "Implement OAuth integration with Google",
  "status": "in_progress",  // Changed from pending
  "priority": "high",
  "id": "google-oauth"
}

// Work completed - mark completed immediately
{
  "content": "Implement OAuth integration with Google", 
  "status": "completed",  // Changed from in_progress
  "priority": "high",
  "id": "google-oauth"
}
```

### Adding Follow-up Tasks
```json
{
  "todos": [
    {
      "content": "Implement OAuth integration with Google",
      "status": "completed",
      "priority": "high", 
      "id": "google-oauth"
    },
    {
      "content": "Add error handling for OAuth failures",  // New task discovered
      "status": "pending",
      "priority": "high",
      "id": "oauth-error-handling"
    },
    {
      "content": "Write integration tests for OAuth flow",  // New task discovered
      "status": "pending",
      "priority": "medium",
      "id": "oauth-integration-tests"
    }
  ]
}
```

### Task Removal
```json
{
  "todos": [
    {
      "content": "Set up Redis for session storage",
      "status": "completed",
      "priority": "high",
      "id": "redis-setup"
    }
    // Task "configure-load-balancer" removed as no longer relevant
  ]
}
```

## Integration Patterns

### Development Workflow
```
1. TodoWrite - Plan implementation tasks
2. Mark first task as in_progress
3. Execute task using appropriate tools
4. Mark task completed immediately
5. Mark next task as in_progress
6. Repeat until all tasks completed
```

### Error Handling Workflow
```
1. Encounter error during task execution
2. Keep current task as in_progress
3. Add new task to investigate/resolve error
4. Resolve error first
5. Return to original task
6. Complete original task
```

### Discovery Workflow
```
1. Start with high-level tasks
2. Mark task as in_progress
3. Discover subtasks during execution
4. Add new tasks to todo list
5. Complete current task
6. Process newly discovered tasks
```

## Best Practices

### Task Organization
1. **Clear Descriptions**: Use specific, actionable language
2. **Logical Ordering**: Arrange tasks in execution order
3. **Dependency Awareness**: Consider task dependencies
4. **Scope Management**: Keep tasks focused and manageable

### Progress Management
```json
// ✅ Good progress tracking
{
  "todos": [
    {
      "content": "Set up CI/CD pipeline with GitHub Actions",
      "status": "completed",
      "priority": "high",
      "id": "cicd-setup"
    },
    {
      "content": "Configure automated testing in pipeline",
      "status": "in_progress",  // Only one in_progress
      "priority": "high",
      "id": "automated-testing"
    },
    {
      "content": "Add deployment automation to staging environment",
      "status": "pending",
      "priority": "medium",
      "id": "staging-deployment"
    }
  ]
}
```

### Task Lifecycle Management
1. **Proactive Creation**: Use when planning complex work
2. **Real-time Updates**: Update status as work progresses  
3. **Immediate Completion**: Mark completed as soon as finished
4. **Cleanup**: Remove irrelevant tasks when discovered
5. **Discovery**: Add new tasks found during implementation

## Performance and User Experience

### User Visibility Benefits
- **Progress Tracking**: Users see work progression
- **Transparency**: Clear view of what's being done
- **Planning Quality**: Demonstrates thorough planning
- **Completion Confidence**: Users know when work is done

### Development Benefits
- **Organization**: Keeps work organized and focused
- **Memory Aid**: Prevents forgetting important tasks
- **Quality Assurance**: Ensures all requirements addressed
- **Methodology**: Encourages systematic approach

### Tool Integration
- **Other Tools**: TodoWrite integrates with all other tools
- **Workflow Management**: Provides structure for complex operations
- **Error Recovery**: Helps manage and track error resolution
- **Feature Development**: Organizes multi-step feature implementation