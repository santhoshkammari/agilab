# Read Tool - Comprehensive Documentation

## Overview
The Read tool provides file reading capabilities with support for text files, images, PDFs, and more. It includes line-based navigation, content truncation handling, and multimodal processing capabilities.

## Function Signature
```json
{
  "name": "Read",
  "parameters": {
    "file_path": "string (required)",
    "offset": "number (optional)",
    "limit": "number (optional)"
  }
}
```

## Parameters

### file_path (required)
- **Type**: string
- **Purpose**: Absolute path to the file to read
- **Constraint**: Must be absolute path (not relative)
- **Format**: `/absolute/path/to/file.ext`
- **Validation**: File must exist and be readable

### offset (optional)
- **Type**: number
- **Purpose**: Line number to start reading from
- **Default**: 1 (beginning of file)
- **Use Case**: Reading specific portions of large files
- **Range**: Positive integers only

### limit (optional)
- **Type**: number
- **Purpose**: Maximum number of lines to read
- **Default**: 2000 lines
- **Use Case**: Controlling memory usage for large files
- **Range**: Positive integers

## File Type Support

### Text Files
- **Extensions**: `.txt`, `.md`, `.js`, `.py`, `.json`, `.xml`, `.csv`, etc.
- **Encoding**: UTF-8 primary support
- **Line Endings**: Handles Unix, Windows, and Mac line endings
- **Processing**: Content returned as-is with line numbers

### Source Code Files
```json
// Programming languages
".js", ".jsx", ".ts", ".tsx"    // JavaScript/TypeScript
".py", ".pyx"                   // Python
".go"                           // Go
".rs"                           // Rust
".java", ".kt"                  // JVM languages
".c", ".cpp", ".h", ".hpp"      // C/C++
".rb"                           // Ruby
".php"                          // PHP
".swift"                        // Swift
".dart"                         // Dart
```

### Configuration Files
```json
// Common config formats
".json", ".yaml", ".yml"        // Data formats
".toml", ".ini", ".cfg"        // Configuration formats
".env"                          // Environment variables
"Dockerfile", "Makefile"        // Build files
".gitignore", ".gitconfig"      // Git files
```

### Image Files
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP, SVG
- **Processing**: Visual content displayed in multimodal interface
- **Use Cases**: Screenshots, diagrams, UI mockups
- **Capability**: Full image analysis and description

### PDF Files
- **Format**: `.pdf`
- **Processing**: Page-by-page text and visual extraction
- **Content**: Both text content and visual elements analyzed
- **Structure**: Maintains document structure and formatting context

### Jupyter Notebooks
- **Format**: `.ipynb`
- **Recommendation**: Use NotebookRead tool instead
- **Fallback**: Can read as JSON but loses notebook-specific formatting

## Content Processing

### Line Number Format
```
     1	First line of content
     2	Second line of content  
     3	Third line of content
```
- **Format**: `spaces + line number + tab + content`
- **Numbering**: Starts at 1, increments by 1
- **Alignment**: Right-aligned line numbers with padding

### Content Truncation
- **Line Limit**: Lines longer than 2000 characters are truncated
- **Indicator**: Truncated lines marked with `...`
- **Preservation**: Line structure maintained even with truncation

### Empty File Handling
```
[System Reminder: File exists but has empty contents]
```
- **Detection**: Empty files specifically identified
- **Warning**: System reminder displayed instead of content
- **Validation**: Confirms file exists vs. file not found

## Usage Patterns

### Basic File Reading
```json
// Read entire file (up to 2000 lines)
{"file_path": "/project/src/app.js"}

// Read from specific line
{"file_path": "/project/README.md", "offset": 10}

// Read limited lines
{"file_path": "/project/large-file.txt", "limit": 50}

// Read specific section
{"file_path": "/project/src/utils.js", "offset": 100, "limit": 30}
```

### Large File Navigation
```json
// Read beginning of large file
{"file_path": "/logs/application.log", "limit": 100}

// Read middle section
{"file_path": "/data/dataset.csv", "offset": 1000, "limit": 500}

// Read end section (requires knowing file length)
{"file_path": "/logs/error.log", "offset": 9500, "limit": 500}
```

### Image Analysis
```json
// Read screenshot for analysis
{"file_path": "/tmp/screenshot.png"}

// Analyze diagram or chart
{"file_path": "/docs/architecture-diagram.png"}

// Read UI mockup
{"file_path": "/designs/homepage-mockup.jpg"}
```

## Multimodal Capabilities

### Image Processing
- **Analysis**: Automatic image description and analysis
- **Text Extraction**: OCR for text within images
- **Element Recognition**: UI elements, diagrams, charts
- **Context Understanding**: Relationship between visual elements

### PDF Processing
- **Text Extraction**: Clean text extraction from PDF pages
- **Visual Analysis**: Images, charts, diagrams within PDFs
- **Structure Recognition**: Headers, paragraphs, lists, tables
- **Page Handling**: Sequential page-by-page processing

### Screenshot Analysis
```json
// Temporary screenshot files
{"file_path": "/var/folders/123/abc/T/TemporaryItems/NSIRD_screencaptureui_ZfB1tD/Screenshot.png"}

// Screen capture analysis
{"file_path": "/tmp/screen-capture-2024-01-15.png"}
```

## Performance Considerations

### Memory Management
- **Default Limit**: 2000 lines prevents memory issues
- **Large Files**: Use offset/limit for efficient reading
- **Streaming**: Tool processes content in chunks
- **Truncation**: Long lines automatically truncated

### Optimization Strategies
```json
// For large files, read in sections
{"file_path": "/large-file.txt", "offset": 1, "limit": 1000}
{"file_path": "/large-file.txt", "offset": 1001, "limit": 1000}

// Preview file structure
{"file_path": "/data-file.csv", "limit": 10}

// Read specific function/class
{"file_path": "/src/component.js", "offset": 45, "limit": 25}
```

### Batch Reading
```json
// Read multiple files efficiently
[
  {"file_path": "/project/package.json"},
  {"file_path": "/project/README.md"},
  {"file_path": "/project/src/index.js"}
]
```

## Error Handling

### Common Errors
1. **File Not Found**: Path doesn't exist
2. **Permission Denied**: Insufficient read permissions
3. **Binary File**: Unsupported binary format
4. **Encoding Issues**: Non-UTF-8 encoding problems
5. **Relative Path**: Using relative instead of absolute path

### Error Response Examples
```json
// File not found
{
  "error": "File not found: /nonexistent/file.txt",
  "code": "FILE_NOT_FOUND"
}

// Permission denied
{
  "error": "Permission denied: /restricted/file.txt", 
  "code": "PERMISSION_DENIED"
}
```

### Recovery Strategies
- **Path Verification**: Check file existence with LS tool
- **Permission Check**: Verify file access rights
- **Format Validation**: Ensure file format is supported
- **Alternative Paths**: Try related or backup files

## Integration Patterns

### Code Analysis Workflow
```json
// 1. Discover files
Glob(pattern="**/*.js")

// 2. Read specific files
Read(file_path="/src/app.js")

// 3. Search for patterns
Grep(pattern="function.*Error", glob="*.js")

// 4. Read matching files
Read(file_path="/src/error-handler.js")
```

### Documentation Review
```json
// Read project documentation
[
  Read(file_path="/README.md"),
  Read(file_path="/CONTRIBUTING.md"),
  Read(file_path="/docs/API.md")
]
```

### Configuration Analysis
```json
// Read configuration files
[
  Read(file_path="/package.json"),
  Read(file_path="/.env.example"),
  Read(file_path="/config/database.json")
]
```

## Best Practices

### Efficient File Reading
1. **Mandatory Read First**: Always read files before editing
2. **Appropriate Limits**: Use limit for large files
3. **Targeted Reading**: Use offset for specific sections
4. **Batch Operations**: Read multiple related files together

### File Type Selection
```json
// Text files - direct read
Read(file_path="/src/app.js")

// Images - multimodal analysis
Read(file_path="/screenshot.png")

// PDFs - structured extraction
Read(file_path="/documentation.pdf")

// Notebooks - specialized tool
NotebookRead(notebook_path="/analysis.ipynb")
```

### Large File Strategies
```json
// Preview approach
{"file_path": "/large-log.txt", "limit": 50}

// Section-by-section
{"file_path": "/large-data.csv", "offset": 1, "limit": 1000}
{"file_path": "/large-data.csv", "offset": 1001, "limit": 1000}

// Targeted reading based on line numbers from grep
{"file_path": "/source.js", "offset": 245, "limit": 20}
```

## Security Considerations

### Access Control
- **Permission Respect**: Honors filesystem permissions
- **Path Validation**: Validates all file paths
- **Symlink Handling**: Follows symbolic links safely
- **Sandboxing**: Operates within available permissions

### Safe Practices
```json
// Safe project file reading
Read(file_path="/home/user/projects/app/src/main.js")

// Temporary file handling
Read(file_path="/tmp/screenshot-12345.png")

// Avoid sensitive files
// Don't read: /etc/passwd, ~/.ssh/id_rsa, etc.
```

### Privacy Considerations
- **Credential Detection**: Be cautious with config files
- **Sensitive Data**: Avoid reading files with secrets
- **Log Analysis**: Be mindful of personally identifiable information
- **Image Content**: Consider privacy implications of screenshot analysis

## Tool Prerequisites

### File Operations Requirement
- **Edit Tool**: Must read file before editing
- **MultiEdit Tool**: Must read file before multi-editing
- **Write Tool**: Must read existing file before overwriting
- **Validation**: Tools will fail if Read not called first

### Integration Requirements
```json
// Required pattern for file modification
1. Read(file_path="/target/file.js")        // Read first
2. Edit(file_path="/target/file.js", ...)   // Then edit
3. Read(file_path="/target/file.js")        // Verify changes (optional)
```