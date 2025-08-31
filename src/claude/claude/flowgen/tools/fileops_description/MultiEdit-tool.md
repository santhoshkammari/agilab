# MultiEdit Tool - Comprehensive Documentation

## Overview
The MultiEdit tool enables multiple find-and-replace operations on a single file in one atomic transaction. Built on the Edit tool, it processes edits sequentially and ensures all changes succeed or none are applied.

## Function Signature
```json
{
  "name": "MultiEdit",
  "parameters": {
    "file_path": "string (required)",
    "edits": "array of edit objects (required, min 1)"
  }
}
```

## Parameters

### file_path (required)
- **Type**: string
- **Purpose**: Absolute path to file to modify
- **Constraint**: Must be absolute path (not relative)
- **Validation**: File must exist and be writable
- **Prerequisite**: Must have used Read tool on file first

### edits (required)
- **Type**: array of edit objects
- **Minimum**: 1 edit object
- **Processing**: Sequential execution in array order
- **Atomicity**: All edits succeed or none are applied

#### Edit Object Structure
```json
{
  "old_string": "string (required)",
  "new_string": "string (required)",
  "replace_all": "boolean (optional, default: false)"
}
```

### Edit Object Parameters

#### old_string (required)
- **Type**: string
- **Purpose**: Exact text to find and replace
- **Matching**: Case-sensitive, whitespace-sensitive exact match
- **Context**: Must match file content exactly

#### new_string (required)
- **Type**: string
- **Purpose**: Replacement text
- **Constraint**: Must be different from corresponding `old_string`
- **Format**: Should maintain code style and indentation

#### replace_all (optional)
- **Type**: boolean
- **Default**: false
- **Purpose**: Replace all occurrences of the string
- **Use Case**: Variable renaming across the file

## Sequential Processing

### Execution Order
1. **First Edit**: Applied to original file content
2. **Second Edit**: Applied to result of first edit
3. **Subsequent Edits**: Applied to cumulative result
4. **Final Result**: All edits applied sequentially

### Important Considerations
- **Order Dependency**: Later edits operate on modified content from earlier edits
- **String Invalidation**: Earlier edits may affect strings targeted by later edits
- **Planning Required**: Carefully plan edit sequence to avoid conflicts

### Example Sequential Processing
```json
{
  "file_path": "/src/app.js",
  "edits": [
    {
      "old_string": "const oldName = 'value';",
      "new_string": "const newName = 'value';"
    },
    {
      "old_string": "console.log(oldName);",
      "new_string": "console.log(newName);"
    }
  ]
}
```

## Atomic Operations

### All-or-Nothing Guarantee
- **Success**: All edits applied successfully
- **Failure**: No edits applied, file remains unchanged
- **Validation**: Each edit validated before any are applied
- **Rollback**: Automatic rollback on any edit failure

### Failure Scenarios
1. **Invalid String**: Any `old_string` not found in (current) file content
2. **Multiple Matches**: Non-unique string without `replace_all: true`
3. **Identical Strings**: Any `old_string` equals corresponding `new_string`
4. **File Issues**: Permission denied, file locked, etc.

## Usage Patterns

### Variable Renaming
```json
{
  "file_path": "/src/component.js",
  "edits": [
    {
      "old_string": "oldVariableName",
      "new_string": "newVariableName",
      "replace_all": true
    },
    {
      "old_string": "// TODO: Update oldVariableName usage",
      "new_string": "// Updated: newVariableName implementation complete"
    }
  ]
}
```

### Function Signature Update
```json
{
  "file_path": "/src/utils.js",
  "edits": [
    {
      "old_string": "function processData(data) {",
      "new_string": "function processData(data, options = {}) {"
    },
    {
      "old_string": "  return data.map(item => transform(item));",
      "new_string": "  return data.map(item => transform(item, options));"
    },
    {
      "old_string": "// Simple data processing function",
      "new_string": "// Data processing function with configurable options"
    }
  ]
}
```

### Configuration Updates
```json
{
  "file_path": "/config/settings.json",
  "edits": [
    {
      "old_string": "  \"version\": \"1.0.0\",",
      "new_string": "  \"version\": \"1.1.0\","
    },
    {
      "old_string": "  \"debug\": true,",
      "new_string": "  \"debug\": false,"
    },
    {
      "old_string": "  \"api_url\": \"http://localhost:3000\"",
      "new_string": "  \"api_url\": \"https://api.production.com\""
    }
  ]
}
```

### Import Statement Reorganization
```json
{
  "file_path": "/src/app.js",
  "edits": [
    {
      "old_string": "import React from 'react';\nimport { useState } from 'react';",
      "new_string": "import React, { useState } from 'react';"
    },
    {
      "old_string": "import './styles.css';\nimport './app.css';",
      "new_string": "import './styles.css';"
    }
  ]
}
```

## File Creation Pattern

### New File Creation
```json
{
  "file_path": "/new/file/path.js",
  "edits": [
    {
      "old_string": "",
      "new_string": "// New file content\nexport const API_BASE = 'https://api.example.com';\n\nexport function fetchData(endpoint) {\n  return fetch(`${API_BASE}/${endpoint}`);\n}"
    }
  ]
}
```

### Template Expansion
```json
{
  "file_path": "/templates/component.jsx",
  "edits": [
    {
      "old_string": "",
      "new_string": "import React from 'react';\n\nconst COMPONENT_NAME = () => {\n  return (\n    <div>\n      {/* Component content */}\n    </div>\n  );\n};\n\nexport default COMPONENT_NAME;"
    },
    {
      "old_string": "COMPONENT_NAME",
      "new_string": "UserProfile",
      "replace_all": true
    }
  ]
}
```

## Advanced Patterns

### Complex Refactoring
```json
{
  "file_path": "/src/legacy-code.js",
  "edits": [
    {
      "old_string": "var oldFunction = function(param) {",
      "new_string": "const newFunction = (param) => {"
    },
    {
      "old_string": "  var result = processParam(param);",
      "new_string": "  const result = await processParamAsync(param);"
    },
    {
      "old_string": "  return result;",
      "new_string": "  return result;"
    },
    {
      "old_string": "};",
      "new_string": "};"
    },
    {
      "old_string": "module.exports = oldFunction;",
      "new_string": "export default newFunction;"
    }
  ]
}
```

### Error Handling Enhancement
```json
{
  "file_path": "/src/api-client.js",
  "edits": [
    {
      "old_string": "try {\n  const response = await fetch(url);\n  return response.json();\n} catch (error) {\n  console.log(error);\n}",
      "new_string": "try {\n  const response = await fetch(url);\n  if (!response.ok) {\n    throw new Error(`HTTP ${response.status}: ${response.statusText}`);\n  }\n  return response.json();\n} catch (error) {\n  logger.error('API request failed:', error);\n  throw error;\n}"
    },
    {
      "old_string": "// Basic error handling",
      "new_string": "// Enhanced error handling with proper HTTP status checking"
    }
  ]
}
```

## Best Practices

### Edit Planning
1. **Read First**: Always read file before planning edits
2. **Order Carefully**: Plan edit sequence to avoid conflicts
3. **Test Individually**: Mentally verify each edit against current state
4. **Atomic Grouping**: Group related changes in single MultiEdit call

### String Selection
```json
// ✅ Good - Specific, context-aware edits
{
  "edits": [
    {
      "old_string": "const API_VERSION = 'v1';",
      "new_string": "const API_VERSION = 'v2';"
    },
    {
      "old_string": "// Using API version v1",
      "new_string": "// Using API version v2"
    }
  ]
}

// ❌ Poor - Generic strings that might conflict
{
  "edits": [
    {
      "old_string": "v1",
      "new_string": "v2",
      "replace_all": true
    }
  ]
}
```

### Conflict Avoidance
```json
// ✅ Safe sequence - no string overlap
{
  "edits": [
    {
      "old_string": "function oldName() {",
      "new_string": "function newName() {"
    },
    {
      "old_string": "return oldName();",
      "new_string": "return newName();"
    }
  ]
}

// ⚠️ Risky - later edit depends on earlier edit not running
{
  "edits": [
    {
      "old_string": "oldName",
      "new_string": "newName",
      "replace_all": true  // This would change second edit's target
    },
    {
      "old_string": "// Function oldName implementation",
      "new_string": "// Function newName implementation"
    }
  ]
}
```

## Error Handling

### Validation Errors
```json
{
  "error": "Edit validation failed",
  "details": {
    "edit_index": 2,
    "reason": "String not found: 'nonexistent string'",
    "file_state": "unchanged"
  }
}
```

### Sequential Processing Failures
```json
{
  "error": "Edit sequence failed at step 3",
  "details": {
    "successful_edits": 2,
    "failed_edit": {
      "old_string": "string_not_found_after_previous_edits",
      "reason": "String not found in modified content"
    },
    "rollback": "complete"
  }
}
```

### Recovery Strategies
1. **Edit Reordering**: Change sequence to avoid conflicts
2. **String Updates**: Update strings based on sequential processing
3. **Scope Reduction**: Break into smaller, safer edit groups
4. **Individual Edits**: Fall back to single Edit tool calls

## Performance Considerations

### Efficiency Factors
- **File Size**: Larger files take longer to process
- **Edit Count**: More edits increase processing time
- **String Complexity**: Complex regex-like strings slower to match
- **File I/O**: Multiple disk operations for atomic guarantee

### Optimization Strategies
```json
// ✅ Efficient - Related edits in one call
{
  "edits": [
    {"old_string": "config.dev", "new_string": "config.prod"},
    {"old_string": "debug: true", "new_string": "debug: false"}
  ]
}

// ❌ Inefficient - Separate MultiEdit calls
// MultiEdit call 1: config.dev -> config.prod  
// MultiEdit call 2: debug: true -> debug: false
```

### File Size Recommendations
- **Small Files** (<10KB): Use MultiEdit freely
- **Medium Files** (10KB-100KB): Group related edits efficiently  
- **Large Files** (>100KB): Consider breaking into smaller edit groups
- **Very Large Files** (>1MB): Use individual Edit calls for critical changes

## Integration Patterns

### Refactoring Workflow
```
1. Read(file_path="/src/component.js")
2. MultiEdit(file_path="/src/component.js", edits=[...])
3. Bash(command="npm run lint /src/component.js")  // Validate syntax
4. Bash(command="npm test component.test.js")      // Validate functionality
```

### Configuration Management
```
1. Read(file_path="/config/app.json")  
2. MultiEdit(file_path="/config/app.json", edits=[...])
3. Bash(command="npm run validate-config")  // Test configuration
```

### Code Migration
```
1. Glob(pattern="**/*.js")  // Find all JavaScript files
2. Read(file_path="/src/each-file.js")  // Read each file
3. MultiEdit(file_path="/src/each-file.js", edits=[...])  // Apply migrations
```

## Tool Comparison

### MultiEdit vs Edit
- **MultiEdit**: Multiple changes to same file, atomic operation
- **Edit**: Single change, simpler validation, faster for one change

### MultiEdit vs Write
- **MultiEdit**: Incremental changes, preserves most content
- **Write**: Complete file replacement, simpler for new files

### MultiEdit vs Notebook Tools
- **MultiEdit**: Text file editing
- **NotebookEdit**: Jupyter notebook cell operations