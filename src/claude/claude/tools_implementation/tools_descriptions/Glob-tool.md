# Glob Tool - Comprehensive Documentation

## Overview
The Glob tool provides fast file pattern matching using glob patterns. It's optimized for performance and works efficiently with codebases of any size, returning matching file paths sorted by modification time.

## Function Signature
```json
{
  "name": "Glob",
  "parameters": {
    "pattern": "string (required)",
    "path": "string (optional)"
  }
}
```

## Parameters

### pattern (required)
- **Type**: string
- **Purpose**: Glob pattern to match files against
- **Syntax**: Standard glob pattern syntax with extensions
- **Performance**: Optimized for large codebases

### path (optional)
- **Type**: string
- **Purpose**: Directory to search in
- **Default**: Current working directory when omitted
- **Validation**: Must be valid directory path if provided
- **Important**: Omit field entirely for default behavior (don't use "undefined" or "null")

## Glob Pattern Syntax

### Basic Patterns
```bash
*.js                    # All JavaScript files in current directory
*.{js,ts}              # All JavaScript and TypeScript files
**/*.py                # All Python files recursively
src/**/*.tsx           # All TypeScript React files in src/
```

### Wildcard Characters
- `*` - Matches any characters except path separator
- `**` - Matches any characters including path separators (recursive)
- `?` - Matches exactly one character
- `[abc]` - Matches any character in brackets
- `[a-z]` - Matches any character in range
- `{js,ts}` - Matches any of the comma-separated alternatives

### Advanced Patterns
```bash
**/*test*.js           # Files containing "test" in name
src/**/[A-Z]*.ts      # TypeScript files starting with uppercase
**/*.{json,yaml,yml}   # Configuration files
**/node_modules/**     # All files in node_modules
```

### Negation Patterns
```bash
**/*.js               # All JS files
!**/node_modules/**   # Exclude node_modules (use with tools that support negation)
!**/*.test.js         # Exclude test files
```

## Performance Characteristics

### Optimization Features
- **Fast Scanning**: Optimized for large directory trees
- **Memory Efficient**: Streams results without loading entire file list
- **Index Aware**: Leverages filesystem indices when available
- **Concurrent Processing**: Parallel directory traversal

### Performance Benchmarks
- **Small Projects** (<1000 files): Sub-millisecond response
- **Medium Projects** (1000-10000 files): <100ms response
- **Large Projects** (>10000 files): <1s response
- **Monorepos**: Efficiently handles massive codebases

### Sorting Behavior
- **Primary Sort**: Modification time (newest first)
- **Secondary Sort**: Alphabetical by path
- **Purpose**: Most recently changed files appear first
- **Use Case**: Prioritizes files likely to be relevant

## Common Usage Patterns

### Language-Specific Searches
```json
// JavaScript/TypeScript
{"pattern": "**/*.{js,jsx,ts,tsx}"}

// Python
{"pattern": "**/*.py"}

// Go
{"pattern": "**/*.go"}

// Rust
{"pattern": "**/*.rs"}

// C/C++
{"pattern": "**/*.{c,cpp,h,hpp}"}

// Configuration files
{"pattern": "**/*.{json,yaml,yml,toml,ini}"}
```

### Project Structure Analysis
```json
// Source code only
{"pattern": "src/**/*"}

// Test files
{"pattern": "**/*{test,spec}*.{js,ts,py}"}

// Documentation
{"pattern": "**/*.{md,rst,txt}"}

// Build artifacts
{"pattern": "**/dist/**/*"}
```

### Framework-Specific Patterns
```json
// React components
{"pattern": "**/*.{jsx,tsx}"}

// Vue components
{"pattern": "**/*.vue"}

// Angular components
{"pattern": "**/*.component.{ts,html,css}"}

// Next.js pages
{"pattern": "**/pages/**/*.{js,ts,jsx,tsx}"}
```

## Return Value Format

### Response Structure
```json
{
  "files": [
    "/absolute/path/to/file1.js",
    "/absolute/path/to/file2.js",
    "/absolute/path/to/file3.js"
  ]
}
```

### Path Format
- **Type**: Array of strings
- **Format**: Absolute file paths
- **Sorting**: By modification time (newest first)
- **Encoding**: UTF-8 encoded paths
- **Platform**: Cross-platform path format

### Empty Results
```json
{
  "files": []
}
```

## Integration with Other Tools

### Workflow Patterns
1. **Discovery → Analysis**:
   ```
   Glob(pattern="**/*.js") → Read(file_path="found_file.js")
   ```

2. **Pattern → Search**:
   ```
   Glob(pattern="**/*.py") → Grep(pattern="class.*Error", glob="*.py")
   ```

3. **Batch Processing**:
   ```
   Glob(pattern="**/*.json") → MultiEdit across all files
   ```

### Speculative Batching
```json
// Search multiple patterns simultaneously
[
  {"pattern": "**/*.js"},
  {"pattern": "**/*.ts"},
  {"pattern": "**/*.py"}
]
```

## Error Handling

### Common Error Conditions
1. **Invalid Pattern**: Malformed glob syntax
2. **Permission Denied**: Insufficient directory access
3. **Path Not Found**: Invalid directory path
4. **Resource Limits**: Pattern too complex or broad

### Error Response Format
```json
{
  "error": "Permission denied: /restricted/path",
  "code": "PERMISSION_DENIED"
}
```

### Recovery Strategies
- **Simplify Pattern**: Use more specific patterns
- **Check Permissions**: Verify directory access
- **Validate Path**: Ensure target directory exists
- **Scope Reduction**: Limit search scope

## Best Practices

### Pattern Construction
1. **Specificity**: Use specific patterns to reduce noise
2. **Performance**: Avoid overly broad patterns like `**/*`
3. **Clarity**: Use descriptive patterns that match intent
4. **Testing**: Test patterns with small scopes first

### When to Use Glob
- **File Discovery**: Finding files by name patterns
- **Bulk Operations**: Preparing file lists for batch processing
- **Project Analysis**: Understanding codebase structure
- **Template Matching**: Finding files following naming conventions

### When NOT to Use Glob
- **Content Search**: Use Grep for searching file contents
- **Single File**: Use Read for known file paths
- **Directory Listing**: Use LS for simple directory contents
- **Complex Logic**: Use Task agent for multi-step searches

## Advanced Features

### Recursive Depth Control
```bash
# All levels (default)
**/*.js

# Limited depth
*/*.js      # One level only
*/*/*.js    # Two levels only
```

### Extension Handling
```bash
# Multiple extensions
*.{js,jsx,ts,tsx,vue}

# No extension
**/Dockerfile
**/Makefile

# Hidden files
**/.*rc
**/.env*
```

### Directory Targeting
```bash
# Specific directories
src/**/*.js
tests/**/*.spec.js
docs/**/*.md

# Exclude patterns (tool-dependent)
**/*.js
!**/node_modules/**
```

## Performance Optimization

### Pattern Efficiency
1. **Front-load Specificity**: Put specific parts early in pattern
2. **Avoid Deep Recursion**: Use targeted directory paths
3. **Batch Related Patterns**: Group similar searches
4. **Cache Results**: Reuse results for related operations

### Resource Management
- **Memory Usage**: Patterns with many matches use more memory
- **Disk I/O**: Broad patterns require more filesystem access
- **CPU Usage**: Complex patterns require more processing
- **Time Limits**: Very broad patterns may hit timeout limits

## Platform Considerations

### Cross-Platform Compatibility
- **Path Separators**: Automatically handles / vs \ 
- **Case Sensitivity**: Respects filesystem case sensitivity
- **Unicode Support**: Full Unicode filename support
- **Symlink Handling**: Follows symbolic links appropriately

### Filesystem-Specific Behavior
- **NTFS**: Windows-specific file attributes respected
- **ext4/APFS**: Linux/Mac optimizations utilized
- **Network Filesystems**: May have different performance characteristics
- **Case-Insensitive**: macOS and Windows case handling

## Security Considerations

### Access Control
- **Directory Permissions**: Respects filesystem permissions
- **Symlink Traversal**: Controlled symlink following
- **Hidden Files**: Can access hidden files if permissions allow
- **System Directories**: May be restricted by system policies

### Safety Features
- **Path Validation**: Validates all path inputs
- **Resource Limits**: Prevents excessive resource usage
- **Sandboxing**: Operates within available permissions
- **Error Isolation**: Errors don't affect other operations