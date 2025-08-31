# Edit Tool - Comprehensive Documentation

## Overview
The Edit tool performs exact string replacements in files. It requires precise string matching and includes safety mechanisms to prevent unintended changes. The tool enforces a read-before-edit policy for safety.

## Function Signature
```json
{
  "name": "Edit",
  "parameters": {
    "file_path": "string (required)",
    "old_string": "string (required)",
    "new_string": "string (required)",
    "replace_all": "boolean (optional, default: false)"
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

### old_string (required)
- **Type**: string
- **Purpose**: Exact text to find and replace
- **Matching**: Case-sensitive, whitespace-sensitive exact match
- **Uniqueness**: Must be unique in file unless `replace_all` is true
- **Format**: Must preserve exact indentation and formatting

### new_string (required)
- **Type**: string
- **Purpose**: Replacement text
- **Constraint**: Must be different from `old_string`
- **Formatting**: Should maintain code style and indentation
- **Content**: Can be empty string for deletion

### replace_all (optional)
- **Type**: boolean
- **Default**: false
- **Purpose**: Replace all occurrences instead of requiring uniqueness
- **Use Case**: Variable renaming, pattern replacement across file

## String Matching Requirements

### Exact Match Precision
The tool requires **exact** string matching including:
- **Whitespace**: Spaces, tabs, newlines must match exactly
- **Indentation**: Leading whitespace must be preserved precisely
- **Case Sensitivity**: Uppercase/lowercase must match exactly
- **Special Characters**: All punctuation, symbols must match

### Line Number Prefix Handling
When copying from Read tool output:
```
    45	    function calculateTotal(items) {
    46	        return items.reduce((sum, item) => sum + item.price, 0);
    47	    }
```

**Correct extraction** (exclude line number prefix):
```json
{
  "old_string": "    function calculateTotal(items) {\n        return items.reduce((sum, item) => sum + item.price, 0);\n    }",
  "new_string": "    function calculateTotal(items) {\n        return items.reduce((sum, item) => sum + item.total, 0);\n    }"
}
```

**Incorrect** (includes line number):
```json
{
  "old_string": "    45	    function calculateTotal(items) {"  // ❌ Includes line number
}
```

### Indentation Preservation
```javascript
// Original code with specific indentation
if (condition) {
    const result = processData(input);
    return result;
}
```

**Correct replacement**:
```json
{
  "old_string": "    const result = processData(input);",
  "new_string": "    const result = await processDataAsync(input);"
}
```

## Safety Mechanisms

### Read-Before-Edit Policy
- **Requirement**: Must use Read tool on file before editing
- **Validation**: Tool will error if no prior read detected
- **Purpose**: Ensures understanding of file content before modification
- **Exception**: None - this is strictly enforced

### Uniqueness Validation
- **Default Behavior**: `old_string` must appear exactly once in file
- **Failure Condition**: Edit fails if string appears 0 or >1 times
- **Override**: Use `replace_all: true` for multiple occurrences
- **Safety**: Prevents unintended bulk changes

### Content Validation
- **String Difference**: `old_string` and `new_string` must be different
- **File Integrity**: Maintains file structure and syntax
- **Encoding**: Preserves file encoding (UTF-8)

## Usage Patterns

### Simple Text Replacement
```json
{
  "file_path": "/src/config.js",
  "old_string": "const API_URL = 'http://localhost:3000';",
  "new_string": "const API_URL = 'https://api.production.com';"
}
```

### Function Modification
```json
{
  "file_path": "/src/utils.js",
  "old_string": "function validateEmail(email) {\n    return email.includes('@');\n}",
  "new_string": "function validateEmail(email) {\n    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n    return emailRegex.test(email);\n}"
}
```

### Variable Renaming (replace_all)
```json
{
  "file_path": "/src/component.js",
  "old_string": "oldVariableName",
  "new_string": "newVariableName",
  "replace_all": true
}
```

### Code Block Replacement
```json
{
  "file_path": "/src/handler.js",
  "old_string": "try {\n    const data = JSON.parse(input);\n    return data;\n} catch (error) {\n    console.log('Parse error');\n    return null;\n}",
  "new_string": "try {\n    const data = JSON.parse(input);\n    return data;\n} catch (error) {\n    logger.error('JSON parse error:', error);\n    throw new ValidationError('Invalid JSON input');\n}"
}
```

### Content Deletion
```json
{
  "file_path": "/src/debug.js",
  "old_string": "console.log('Debug info:', data);",
  "new_string": ""
}
```

## Error Conditions

### String Not Found
```json
{
  "error": "String not found in file",
  "details": "The specified old_string does not exist in the file"
}
```

### Multiple Matches (without replace_all)
```json
{
  "error": "Multiple matches found",
  "details": "Found 3 occurrences. Use replace_all: true to replace all"
}
```

### File Not Read First
```json
{
  "error": "File must be read before editing",
  "details": "Use Read tool on file before attempting edits"
}
```

### Identical Strings
```json
{
  "error": "old_string and new_string are identical",
  "details": "No changes would be made"
}
```

## Advanced Usage

### Multi-line String Replacement
```json
{
  "file_path": "/src/component.jsx",
  "old_string": "const Component = () => {\n  return (\n    <div>\n      <h1>Title</h1>\n    </div>\n  );\n};",
  "new_string": "const Component = ({ title }) => {\n  return (\n    <div>\n      <h1>{title}</h1>\n    </div>\n  );\n};"
}
```

### Configuration Updates
```json
{
  "file_path": "/package.json",
  "old_string": "  \"version\": \"1.0.0\",",
  "new_string": "  \"version\": \"1.1.0\","
}
```

### Import Statement Changes
```json
{
  "file_path": "/src/app.js",
  "old_string": "import { oldFunction } from './utils';",
  "new_string": "import { newFunction } from './utils';"
}
```

### Bulk Variable Renaming
```json
{
  "file_path": "/src/calculator.js",
  "old_string": "oldVarName",
  "new_string": "newVarName",
  "replace_all": true
}
```

## Best Practices

### Preparation Workflow
1. **Read First**: Always use Read tool before editing
2. **Identify Target**: Locate exact string to replace
3. **Copy Precisely**: Copy exact text including whitespace
4. **Plan Change**: Design replacement maintaining code style
5. **Execute Edit**: Perform single, focused edit
6. **Verify**: Optionally read file again to confirm changes

### String Selection Strategies
```json
// ✅ Good - Specific, unique string
{
  "old_string": "function calculateTax(amount, rate) {"
}

// ❌ Poor - Too generic, likely multiple matches
{
  "old_string": "function"
}

// ✅ Good - Include enough context for uniqueness
{
  "old_string": "  const tax = amount * rate;\n  return tax;"
}
```

### Context-Aware Replacement
```json
// Include surrounding context for uniqueness
{
  "old_string": "// Calculate user preferences\nconst preferences = getDefaultPreferences();",
  "new_string": "// Calculate user preferences\nconst preferences = await getUserPreferences(userId);"
}
```

## Integration Patterns

### Code Refactoring Workflow
```
1. Read(file_path="/src/component.js")
2. Edit(file_path="/src/component.js", old_string="...", new_string="...")
3. Read(file_path="/src/component.js")  // Verify changes
```

### Configuration Management
```
1. Read(file_path="/config/app.json")
2. Edit(file_path="/config/app.json", old_string="...", new_string="...")
3. Bash(command="npm run validate-config")  // Test configuration
```

### Bug Fix Workflow
```
1. Grep(pattern="buggy_function", output_mode="files_with_matches")
2. Read(file_path="/src/identified_file.js")
3. Edit(file_path="/src/identified_file.js", old_string="...", new_string="...")
```

## Performance Considerations

### File Size Impact
- **Small Files** (<10KB): Near-instantaneous edits
- **Medium Files** (10KB-1MB): Fast processing
- **Large Files** (>1MB): May take longer, consider MultiEdit for batch changes

### Optimization Strategies
1. **Specific Strings**: Use unique, specific strings to avoid replace_all
2. **Single Edits**: Prefer single Edit over multiple small edits
3. **Batch Planning**: Use MultiEdit for multiple changes to same file
4. **Context Minimization**: Include just enough context for uniqueness

## Security and Safety

### Change Validation
- **Syntax Preservation**: Ensure edits maintain valid syntax
- **Logic Integrity**: Verify changes don't break program logic
- **Test Coverage**: Run tests after significant edits
- **Version Control**: Commit changes incrementally

### Safe Editing Practices
```json
// ✅ Safe - Specific, surgical change
{
  "old_string": "const DEBUG = true;",
  "new_string": "const DEBUG = false;"
}

// ⚠️ Risky - Very broad change
{
  "old_string": "true",
  "new_string": "false",
  "replace_all": true
}
```

### Rollback Considerations
- **Version Control**: Ensure files are under version control
- **Backup Strategy**: Consider file backups for critical changes
- **Incremental Changes**: Make small, easily reversible edits
- **Testing**: Validate changes immediately after editing