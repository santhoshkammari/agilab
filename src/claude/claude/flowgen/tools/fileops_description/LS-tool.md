# LS Tool - Comprehensive Documentation

## Overview
The LS tool lists files and directories in a specified path with optional glob-based filtering. It provides directory exploration capabilities with ignore patterns and works exclusively with absolute paths.

## Function Signature
```json
{
  "name": "LS",
  "parameters": {
    "path": "string (required)",
    "ignore": "array of strings (optional)"
  }
}
```

## Parameters

### path (required)
- **Type**: string
- **Purpose**: Directory to list contents of
- **Constraint**: Must be absolute path (not relative)
- **Format**: `/absolute/path/to/directory`
- **Validation**: Path must exist and be accessible

### ignore (optional)
- **Type**: array of strings
- **Purpose**: Glob patterns to exclude from listing
- **Format**: Array of glob pattern strings
- **Examples**: `["node_modules", "*.log", ".git"]`

## Path Requirements

### Absolute Path Mandatory
```json
// ✅ Correct - absolute path
{"path": "/home/user/project"}

// ❌ Incorrect - relative path  
{"path": "./project"}
{"path": "../parent"}
{"path": "subfolder"}
```

### Platform-Specific Paths
```json
// Linux/macOS
{"path": "/home/username/documents"}
{"path": "/var/log"}
{"path": "/usr/local/bin"}

// Windows (if applicable)
{"path": "C:\\Users\\Username\\Documents"}
{"path": "D:\\Projects\\MyApp"}
```

## Ignore Patterns

### Basic Ignore Patterns
```json
{
  "path": "/project/root",
  "ignore": [
    "node_modules",     // Exact directory name
    ".git",             // Hidden git directory
    "*.log",            // All log files
    "dist",             // Build output directory
    "coverage"          // Test coverage directory
  ]
}
```

### Advanced Glob Patterns
```json
{
  "path": "/src",
  "ignore": [
    "**/*.test.js",     // Test files recursively
    "**/node_modules",  // node_modules at any level
    ".*",               // All hidden files/directories
    "*.{tmp,temp}",     // Temporary files
    "**/*.{log,cache}"  // Log and cache files
  ]
}
```

### Development Environment Ignores
```json
{
  "path": "/workspace",
  "ignore": [
    // Dependencies
    "node_modules",
    "__pycache__",
    ".venv",
    "vendor",
    
    // Build outputs
    "dist",
    "build",
    "target",
    "bin",
    
    // IDE files
    ".vscode",
    ".idea", 
    "*.swp",
    
    // OS files
    ".DS_Store",
    "Thumbs.db"
  ]
}
```

## Return Value Format

### Basic Directory Listing
```json
{
  "entries": [
    {
      "name": "package.json",
      "type": "file",
      "size": 1024,
      "modified": "2024-01-15T10:30:00Z"
    },
    {
      "name": "src",
      "type": "directory", 
      "size": null,
      "modified": "2024-01-15T09:15:00Z"
    },
    {
      "name": "README.md",
      "type": "file",
      "size": 2048,
      "modified": "2024-01-14T16:45:00Z"
    }
  ]
}
```

### Entry Properties
- **name**: File or directory name (string)
- **type**: `"file"` or `"directory"` (string)
- **size**: File size in bytes (number) or `null` for directories
- **modified**: Last modification timestamp (ISO 8601 string)

### Empty Directory
```json
{
  "entries": []
}
```

## Use Cases

### Project Exploration
```json
// Explore project root
{
  "path": "/project",
  "ignore": ["node_modules", ".git", "dist"]
}

// Check source directory structure
{
  "path": "/project/src",
  "ignore": ["*.test.js", "*.spec.js"]
}
```

### Pre-operation Verification
```json
// Before creating subdirectories
{
  "path": "/parent/directory"
}

// Before file operations
{
  "path": "/target/location",
  "ignore": ["*.tmp"]
}
```

### Configuration Analysis
```json
// Find configuration files
{
  "path": "/project",
  "ignore": ["node_modules", "src", "tests"]
}

// Check config directories
{
  "path": "/etc/myapp"
}
```

## Tool Selection Guidelines

### When to Use LS
- **Directory Contents**: Simple directory listing needs
- **Pre-operation Checks**: Verify directories before file operations
- **Structure Verification**: Confirm expected directory structure
- **Path Validation**: Check if directories exist and are accessible

### When to Use Alternative Tools
- **File Pattern Matching**: Use Glob for finding files by patterns
- **Content Search**: Use Grep for searching inside files  
- **Recursive Search**: Use Glob with `**/*` patterns
- **Complex Queries**: Use Task agent for multi-step directory analysis

## Integration Patterns

### Pre-operation Workflow
```json
// 1. Verify target directory exists
{"path": "/target/directory"}

// 2. Perform operation (e.g., create files)
// 3. Verify results
{"path": "/target/directory"}
```

### Structure Validation
```json
// Check expected project structure
[
  {"path": "/project"},
  {"path": "/project/src"}, 
  {"path": "/project/tests"},
  {"path": "/project/docs"}
]
```

### Configuration Discovery
```json
// Find configuration in common locations
[
  {"path": "/etc/myapp"},
  {"path": "/home/user/.config/myapp"},
  {"path": "/project/config"}
]
```

## Error Handling

### Common Errors
1. **Path Not Found**: Directory doesn't exist
2. **Permission Denied**: Insufficient access rights
3. **Not a Directory**: Path points to file, not directory
4. **Relative Path**: Using relative instead of absolute path

### Error Response Format
```json
{
  "error": "Path not found: /nonexistent/directory",
  "code": "PATH_NOT_FOUND"
}
```

### Recovery Strategies
- **Path Validation**: Check parent directories exist
- **Permission Check**: Verify access rights
- **Path Correction**: Ensure absolute path format
- **Alternative Paths**: Try related directory paths

## Performance Considerations

### Optimization Strategies
1. **Selective Ignoring**: Use ignore patterns to reduce listing size
2. **Shallow Listings**: List specific directories rather than broad scans
3. **Batch Operations**: Combine multiple LS calls when possible
4. **Cache Awareness**: Directory contents may change between calls

### Large Directory Handling
```json
// For large directories, use focused ignores
{
  "path": "/large/directory",
  "ignore": [
    "**/*.log",
    "**/cache/**",
    "**/temp/**",
    "**/*.tmp"
  ]
}
```

## Security Considerations

### Access Control
- **Permission Respect**: Honors filesystem permissions
- **Symlink Handling**: Follows symbolic links appropriately
- **Hidden Files**: Can list hidden files/directories if accessible
- **System Directories**: May be restricted by system policies

### Safe Practices
```json
// Safe project exploration
{
  "path": "/home/user/projects/myapp",
  "ignore": [".env", "*.key", "secrets.*"]
}

// Avoid sensitive directories
// Don't use: /etc/shadow, /root, etc.
```

## Platform Differences

### Linux/macOS
```json
// Standard Unix paths
{"path": "/home/username"}
{"path": "/var/log"}
{"path": "/usr/local/bin"}
```

### Windows Compatibility
```json
// Windows-style paths (if supported)
{"path": "C:\\Users\\Username"}
{"path": "D:\\Projects"}
```

### Cross-Platform Considerations
- **Path Separators**: Tool handles platform differences
- **Case Sensitivity**: Respects filesystem case rules
- **Permission Models**: Works within platform security model
- **Special Characters**: Handles Unicode and special characters

## Best Practices

### Efficient Directory Exploration
1. **Start Specific**: Begin with targeted directory paths
2. **Use Ignores**: Filter out irrelevant content early
3. **Incremental Exploration**: Drill down based on findings
4. **Validate First**: Check directory existence before operations

### Pattern Management
```json
// Reusable ignore patterns for different contexts
const commonIgnores = [
  "node_modules", ".git", "dist", "build",
  "*.log", "*.tmp", ".DS_Store"
];

// Development ignores
const devIgnores = [
  ...commonIgnores,
  "coverage", "*.test.js", ".vscode"
];

// Production ignores  
const prodIgnores = [
  ...commonIgnores,
  "src", "tests", "docs"
];
```

### Integration Workflows
1. **Verification Pattern**: LS → Operation → LS (verify)
2. **Discovery Pattern**: LS → Glob → Grep (explore)
3. **Validation Pattern**: LS multiple paths → aggregate results
4. **Monitoring Pattern**: Periodic LS to detect changes