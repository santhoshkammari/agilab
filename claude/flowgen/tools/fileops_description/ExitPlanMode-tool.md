# ExitPlanMode Tool - Comprehensive Documentation

## Overview
The ExitPlanMode tool is used to transition from planning mode to implementation mode. It presents a structured plan to the user for approval before beginning actual code implementation. This tool ensures clear communication about implementation steps before execution begins.

## Function Signature
```json
{
  "name": "ExitPlanMode",
  "parameters": {
    "plan": "string (required)"
  }
}
```

## Parameters

### plan (required)
- **Type**: string
- **Purpose**: Markdown-formatted implementation plan for user approval
- **Format**: Supports full markdown formatting
- **Content**: Concise but comprehensive plan outline
- **Structure**: Should be organized and easy to understand

## When to Use ExitPlanMode

### Required Scenarios
**ONLY** use this tool when:
1. **Planning Implementation Steps**: Task requires planning the implementation steps of a coding task
2. **Code Writing Required**: The task involves writing actual code
3. **Implementation Planning**: You need to plan how to implement features, fixes, or enhancements
4. **After Planning**: You have finished presenting your plan and are ready to code

### When NOT to Use
**DO NOT** use this tool for:
1. **Research Tasks**: Gathering information, searching files, reading files
2. **Understanding Tasks**: General codebase understanding or exploration
3. **Analysis Tasks**: Code analysis, documentation review, investigation
4. **Information Gathering**: Learning about existing implementations

### Task Type Examples

#### ✅ Use ExitPlanMode
```
User: "Help me implement yank mode for vim"
Assistant: [Plans implementation steps] → Use ExitPlanMode

User: "Add dark mode toggle to the application"  
Assistant: [Plans feature implementation] → Use ExitPlanMode

User: "Fix the memory leak in the data processing module"
Assistant: [Plans debugging and fix approach] → Use ExitPlanMode
```

#### ❌ Do NOT Use ExitPlanMode
```
User: "Search for and understand the implementation of vim mode in the codebase"
Assistant: [Searches and analyzes code] → Do NOT use ExitPlanMode

User: "What authentication methods are used in this project?"
Assistant: [Investigates and reports] → Do NOT use ExitPlanMode

User: "Find all error handling patterns in the codebase"
Assistant: [Searches and documents findings] → Do NOT use ExitPlanMode
```

## Plan Structure

### Effective Plan Format
```markdown
## Implementation Plan: Feature Name

### Overview
Brief description of what will be implemented.

### Steps
1. **Step 1**: Specific action with clear deliverable
2. **Step 2**: Next action building on previous step  
3. **Step 3**: Final implementation details

### Technical Approach
- Key technical decisions
- Libraries or frameworks to use
- Architecture considerations

### Testing Strategy
- How the implementation will be tested
- Test cases to be written

### Acceptance Criteria
- Clear success metrics
- What defines completion
```

### Plan Content Guidelines
1. **Concise**: Plans should be comprehensive but not verbose
2. **Actionable**: Each step should be clear and executable
3. **Ordered**: Steps should follow logical implementation order
4. **Testable**: Include validation and testing considerations
5. **Complete**: Cover all aspects needed for successful implementation

## Usage Examples

### Feature Implementation Plan
```json
{
  "plan": "## Implementation Plan: User Authentication System\n\n### Overview\nImplement secure user authentication with JWT tokens, including registration, login, logout, and password reset functionality.\n\n### Steps\n1. **Database Setup**: Create user table with password hashing\n2. **Auth Middleware**: Implement JWT token validation middleware\n3. **Registration Endpoint**: Create user registration with email validation\n4. **Login System**: Build login endpoint with rate limiting\n5. **Password Reset**: Implement secure password reset flow\n6. **Frontend Integration**: Connect auth system to React components\n\n### Technical Approach\n- Use bcrypt for password hashing\n- JWT tokens with 24-hour expiration\n- Express.js middleware for route protection\n- Email service for password reset\n\n### Testing Strategy\n- Unit tests for auth functions\n- Integration tests for endpoints\n- Security testing for common vulnerabilities\n\n### Acceptance Criteria\n- Users can register, login, and logout securely\n- Password reset works via email\n- All auth routes are properly protected\n- Tests achieve >90% coverage"
}
```

### Bug Fix Plan
```json
{
  "plan": "## Bug Fix Plan: Memory Leak in Data Processing\n\n### Problem Analysis\nIdentified memory leak in event listeners that aren't properly cleaned up when components unmount.\n\n### Steps\n1. **Reproduce Issue**: Set up test environment to consistently reproduce the leak\n2. **Profile Memory**: Use Chrome DevTools to identify leak sources\n3. **Fix Event Listeners**: Implement proper cleanup in useEffect hooks\n4. **Add Cleanup Tests**: Create tests to prevent regression\n5. **Verify Fix**: Confirm memory usage returns to baseline\n\n### Technical Approach\n- Use React useEffect cleanup functions\n- Implement AbortController for fetch requests\n- Add memory monitoring to CI pipeline\n\n### Testing Strategy\n- Memory profiling tests\n- Component mount/unmount stress tests\n- Production environment validation\n\n### Success Criteria\n- Memory usage stable after component operations\n- No memory growth in production monitoring\n- All tests pass including new memory tests"
}
```

### Refactoring Plan
```json
{
  "plan": "## Refactoring Plan: Convert Class Components to Hooks\n\n### Scope\nRefactor legacy class components to modern React hooks for better performance and maintainability.\n\n### Steps\n1. **Audit Components**: Identify all class components requiring conversion\n2. **Convert State Logic**: Transform this.state to useState hooks\n3. **Migrate Lifecycle Methods**: Convert to useEffect patterns\n4. **Update Tests**: Modify tests for hook-based components\n5. **Performance Validation**: Ensure no performance regressions\n\n### Technical Approach\n- Use React 18 hooks (useState, useEffect, useCallback, useMemo)\n- Maintain existing component APIs\n- Preserve all current functionality\n- Follow React hooks best practices\n\n### Testing Strategy\n- Component behavior tests remain unchanged\n- Add new tests for hook-specific logic\n- Performance benchmarking before/after\n\n### Success Metrics\n- All components converted successfully\n- No functional regressions\n- Test coverage maintained or improved\n- Bundle size reduction achieved"
}
```

## User Interaction Flow

### Tool Behavior
1. **Plan Presentation**: Tool displays the implementation plan to user
2. **User Prompt**: System prompts user to approve plan before proceeding
3. **Implementation Start**: Upon approval, begin actual implementation
4. **Plan Visibility**: User sees complete plan before work begins

### User Experience
```
Assistant: [Analyzes requirements and creates plan]
Assistant: [Calls ExitPlanMode with detailed plan]
System: [Shows plan to user and asks for approval]
User: "Looks good, proceed" or "Please modify X in the plan"
Assistant: [Begins implementation based on approved plan]
```

## Integration with Planning Workflow

### Complete Planning Process
```
1. User provides implementation request
2. Assistant analyzes requirements
3. Assistant creates implementation plan
4. Assistant calls ExitPlanMode(plan="...")
5. User reviews and approves plan
6. Assistant begins actual implementation
7. Assistant follows planned steps systematically
```

### Plan Modification Workflow
```
1. ExitPlanMode presents initial plan
2. User requests modifications
3. Assistant revises plan based on feedback
4. Assistant calls ExitPlanMode with updated plan
5. User approves revised plan
6. Implementation begins
```

## Best Practices

### Plan Quality
1. **Clarity**: Use clear, unambiguous language
2. **Completeness**: Cover all necessary implementation aspects
3. **Feasibility**: Ensure plan is realistic and achievable
4. **Testability**: Include validation and testing considerations

### User Communication
```markdown
## ✅ Good Plan Structure
### Overview
Clear problem statement and solution approach

### Implementation Steps
1. Specific, actionable steps
2. Clear dependencies and order
3. Measurable deliverables

### Technical Details
- Architecture decisions explained
- Technology choices justified
- Integration considerations covered

### Validation
- Testing approach defined
- Success criteria specified
```

### Tool Timing
1. **After Analysis**: Use only after thorough requirement analysis
2. **Before Coding**: Always use before beginning implementation
3. **Plan Completion**: Use when plan is complete and ready for approval
4. **Not for Research**: Never use during information gathering phases

## Error Prevention

### Common Mistakes
```json
// ❌ Wrong - Using for research task
User: "Understand how authentication works in this app"
// Should NOT use ExitPlanMode - this is analysis, not implementation

// ✅ Correct - Using for implementation task  
User: "Add OAuth authentication to this app"
// Should use ExitPlanMode after planning implementation steps
```

### Task Classification Guidelines
- **Implementation Tasks**: Require building, fixing, or modifying code → Use ExitPlanMode
- **Research Tasks**: Require understanding, finding, or analyzing → Do NOT use ExitPlanMode
- **Hybrid Tasks**: Analyze first (no ExitPlanMode), then implement (use ExitPlanMode)

## Integration with Other Tools

### Planning Phase (before ExitPlanMode)
- **Read**: Understand existing code structure
- **Grep**: Find related implementations  
- **Glob**: Discover relevant files
- **Task**: Research complex requirements

### Implementation Phase (after ExitPlanMode)
- **Write**: Create new files
- **Edit/MultiEdit**: Modify existing files
- **Bash**: Run build/test commands
- **TodoWrite**: Track implementation progress

## Tool Limitations

### Scope Boundaries
- **Single Use**: Typically used once per implementation task
- **Plan Focus**: Only for actual implementation planning
- **User Dependent**: Requires user approval to proceed
- **No Code Execution**: Tool itself doesn't implement anything

### Context Requirements
- **Clear Requirements**: Needs well-defined implementation requirements
- **Technical Understanding**: Requires understanding of technical approach
- **Complete Planning**: Should have thorough plan before using
- **Implementation Ready**: Only use when ready to begin coding

## Success Indicators

### Effective Plan Characteristics
1. **User Approval**: User approves plan without major modifications
2. **Implementation Success**: Plan leads to successful implementation
3. **Comprehensive Coverage**: All requirements addressed in plan
4. **Clear Communication**: User understands what will be implemented

### Plan Quality Metrics
- **Clarity**: Steps are unambiguous and actionable
- **Completeness**: All implementation aspects covered
- **Feasibility**: Plan is realistic given constraints
- **Testability**: Success can be measured and validated