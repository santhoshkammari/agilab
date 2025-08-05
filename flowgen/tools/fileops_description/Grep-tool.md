# Grep Tool - Comprehensive Documentation

## Overview
The Grep tool is a powerful search engine built on ripgrep (rg) that provides fast, regex-based content searching across files. It offers multiple output modes, advanced filtering, and optimized performance for large codebases.

## Function Signature
```json
{
  "name": "Grep",
  "parameters": {
    "pattern": "string (required)",
    "path": "string (optional)",
    "output_mode": "enum (optional)",
    "glob": "string (optional)", 
    "type": "string (optional)",
    "-i": "boolean (optional)",
    "-n": "boolean (optional)",
    "-A": "number (optional)",
    "-B": "number (optional)", 
    "-C": "number (optional)",
    "multiline": "boolean (optional)",
    "head_limit": "number (optional)"
  }
}
```

## Parameters

### pattern (required)
- **Type**: string
- **Purpose**: Regular expression pattern to search for
- **Engine**: ripgrep (not standard grep)
- **Syntax**: Full regex support with ripgrep extensions
- **Escaping**: Literal braces need escaping (`interface\\{\\}` for Go `interface{}`)

### path (optional)
- **Type**: string  
- **Purpose**: File or directory to search in
- **Default**: Current working directory
- **Format**: Absolute or relative paths accepted

### output_mode (optional)
- **Type**: enum
- **Values**: `"content"` | `"files_with_matches"` | `"count"`
- **Default**: `"files_with_matches"`
- **Behavior**:
  - `content`: Shows matching lines with context
  - `files_with_matches`: Shows only file paths containing matches
  - `count`: Shows match counts per file

### File Filtering

#### glob (optional)
- **Type**: string
- **Purpose**: Filter files using glob patterns
- **Examples**: `"*.js"`, `"*.{ts,tsx}"`, `"**/*.py"`
- **Performance**: More efficient than regex filtering

#### type (optional)
- **Type**: string
- **Purpose**: Filter by file type (ripgrep built-in types)
- **Common Types**: `js`, `py`, `rust`, `go`, `java`, `cpp`, `html`, `css`
- **Efficiency**: Faster than glob for standard file types

### Search Modifiers

#### -i (optional)
- **Type**: boolean
- **Purpose**: Case insensitive search
- **Default**: false (case sensitive)

#### multiline (optional)
- **Type**: boolean
- **Purpose**: Enable multiline mode where `.` matches newlines
- **Default**: false
- **Use Case**: Patterns spanning multiple lines
- **Flags**: Enables `rg -U --multiline-dotall`

### Context Options (content mode only)

#### -n (optional)
- **Type**: boolean
- **Purpose**: Show line numbers in output
- **Requires**: `output_mode: "content"`

#### -A (optional)
- **Type**: number
- **Purpose**: Lines of context after each match
- **Requires**: `output_mode: "content"`

#### -B (optional)
- **Type**: number
- **Purpose**: Lines of context before each match
- **Requires**: `output_mode: "content"`

#### -C (optional)
- **Type**: number
- **Purpose**: Lines of context before and after each match
- **Requires**: `output_mode: "content"`

### Output Limiting

#### head_limit (optional)
- **Type**: number
- **Purpose**: Limit output to first N entries
- **Behavior**:
  - `content` mode: Limits output lines
  - `files_with_matches` mode: Limits file paths
  - `count` mode: Limits count entries
- **Default**: No limit (shows all results)

## Regex Pattern Syntax

### Basic Patterns
```regex
error                    # Literal text
function.*Error          # Function followed by Error
log.*Error              # Log followed by Error
\berror\b               # Word boundary match
```

### Character Classes
```regex
[abc]                   # Any of a, b, c
[a-z]                   # Any lowercase letter
[0-9]                   # Any digit
\d                      # Digit shorthand
\w                      # Word character
\s                      # Whitespace
```

### Quantifiers
```regex
error?                  # Optional 'r'
error+                  # One or more 'r'
error*                  # Zero or more 'r'
error{3}                # Exactly 3 'r's
error{2,5}              # 2 to 5 'r's
```

### Anchors
```regex
^error                  # Start of line
error$                  # End of line
\berror\b               # Word boundaries
```

### Special Cases
```regex
interface\\{\\}         # Literal braces (Go interface{})
"[^"]*"                 # Quoted strings
\/\*.*?\*\/             # C-style comments (single line)
```

## Output Modes

### files_with_matches (default)
```json
{
  "pattern": "function.*Error",
  "output_mode": "files_with_matches"
}
```
**Returns**: Array of file paths
```
/src/app.js
/src/utils.js
/tests/error.test.js
```

### content
```json
{
  "pattern": "function.*Error", 
  "output_mode": "content",
  "-n": true,
  "-C": 2
}
```
**Returns**: Matching lines with context
```
/src/app.js:45:  console.log('Starting...');
/src/app.js:46:  
/src/app.js:47:  function handleError(err) {
/src/app.js:48:    console.error('Error occurred:', err);
/src/app.js:49:  }
```

### count
```json
{
  "pattern": "TODO",
  "output_mode": "count"
}
```
**Returns**: Match counts per file
```
/src/app.js:3
/src/utils.js:7
/src/components/Button.js:1
```

## Advanced Search Patterns

### Code Structure Searches
```json
// Function definitions
{"pattern": "function\\s+\\w+", "type": "js"}

// Class definitions  
{"pattern": "class\\s+\\w+", "type": "py"}

// Import statements
{"pattern": "^import\\s+", "type": "js"}

// Error handling
{"pattern": "try\\s*\\{|catch\\s*\\(", "type": "js"}
```

### Configuration Searches
```json
// Environment variables
{"pattern": "process\\.env\\.", "type": "js"}

// API endpoints
{"pattern": "https?://[^\\s\"']+", "glob": "*.{js,ts,py}"}

// Database queries
{"pattern": "SELECT|INSERT|UPDATE|DELETE", "-i": true}
```

### Documentation Searches
```json
// TODO comments
{"pattern": "TODO|FIXME|HACK", "-i": true}

// Function documentation
{"pattern": "/\\*\\*[\\s\\S]*?\\*/", "multiline": true}

// Markdown headers
{"pattern": "^#{1,6}\\s+", "glob": "*.md"}
```

## Performance Optimization

### Search Strategy
1. **Type First**: Use `type` parameter for standard file types
2. **Glob Secondary**: Use `glob` for custom file filtering  
3. **Limit Results**: Use `head_limit` for large result sets
4. **Scope Path**: Specify narrow paths when possible

### Pattern Efficiency
```json
// Efficient - specific pattern
{"pattern": "^function\\s+handleError", "type": "js"}

// Less efficient - broad pattern
{"pattern": ".*error.*", "-i": true}

// Efficient - bounded search
{"pattern": "class\\s+\\w+Error", "glob": "**/*.py"}
```

### Batching Strategies
```json
// Batch multiple related searches
[
  {"pattern": "class.*Error", "type": "py", "output_mode": "files_with_matches"},
  {"pattern": "def.*error", "type": "py", "output_mode": "files_with_matches"},
  {"pattern": "raise\\s+\\w+Error", "type": "py", "output_mode": "files_with_matches"}
]
```

## Integration Patterns

### Discovery Workflow
1. **Find Files**: Use `files_with_matches` mode
2. **Examine Content**: Use `content` mode with context
3. **Count Occurrences**: Use `count` mode for metrics

### Multi-step Analysis
```json
// Step 1: Find files with errors
{"pattern": "error", "output_mode": "files_with_matches", "type": "js"}

// Step 2: Examine specific error patterns
{"pattern": "throw new \\w+Error", "output_mode": "content", "-n": true, "type": "js"}

// Step 3: Count error types
{"pattern": "\\w+Error", "output_mode": "count", "type": "js"}
```

### Integration with Other Tools
```
Grep → Read: Find files, then read specific files
Grep → Edit: Find patterns, then edit matches
Glob → Grep: Find files by name, then search content
```

## Error Handling

### Common Errors
1. **Invalid Regex**: Malformed pattern syntax
2. **Path Not Found**: Invalid search path
3. **Permission Denied**: Insufficient file access
4. **Resource Limits**: Pattern too complex or broad

### Error Recovery
```json
// Simplify complex patterns
{"pattern": "simple_pattern", "type": "js"}

// Add file type filtering
{"pattern": "broad_pattern", "type": "specific_type"}

// Limit scope
{"pattern": "pattern", "path": "/specific/directory"}
```

## Use Cases

### Code Analysis
- **Function Discovery**: Find function definitions
- **API Usage**: Locate API calls and imports
- **Error Handling**: Identify error patterns
- **Code Quality**: Find TODO/FIXME comments

### Security Analysis
- **Credential Search**: Find hardcoded secrets
- **Vulnerability Patterns**: Locate insecure code
- **Input Validation**: Find user input handling
- **Authentication**: Locate auth-related code

### Refactoring Support
- **Usage Search**: Find all usages of functions/variables
- **Pattern Migration**: Locate old patterns for updating
- **Dependency Analysis**: Find import/require statements
- **Dead Code**: Identify unused code patterns

## Best Practices

### Pattern Construction
1. **Specific First**: Start with specific patterns, broaden if needed
2. **Escape Properly**: Escape special regex characters in code patterns
3. **Test Incrementally**: Test patterns on small scope first
4. **Context Aware**: Use context options for better understanding

### Performance Guidelines
1. **File Filtering**: Always use `type` or `glob` when possible
2. **Result Limiting**: Use `head_limit` for exploratory searches
3. **Path Scoping**: Search specific directories when known
4. **Output Mode**: Choose appropriate output mode for task

### Tool Selection
- **Content Search**: Use Grep for searching inside files
- **File Finding**: Use Glob for finding files by name
- **Open-ended Search**: Use Task agent for multi-step searches
- **Specific Files**: Use Read for examining known files