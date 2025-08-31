# Write Tool - Comprehensive Documentation

## Overview
The Write tool creates or completely overwrites files with new content. It enforces a read-before-write policy for existing files and includes safety mechanisms to prevent accidental data loss.

## Function Signature
```json
{
  "name": "Write",
  "parameters": {
    "file_path": "string (required)",
    "content": "string (required)"
  }
}
```

## Parameters

### file_path (required)
- **Type**: string
- **Purpose**: Absolute path to file to create or overwrite
- **Constraint**: Must be absolute path (not relative)
- **Behavior**: Creates file if it doesn't exist, overwrites if it does
- **Validation**: Parent directory must exist

### content (required)
- **Type**: string
- **Purpose**: Complete file content to write
- **Encoding**: UTF-8 text content
- **Size**: No explicit limit, but consider memory constraints
- **Format**: Raw text content including newlines, indentation

## Safety Mechanisms

### Read-Before-Write Policy
- **Existing Files**: Must use Read tool on existing files before overwriting
- **New Files**: No read required for non-existent files
- **Validation**: Tool will error if existing file not read first
- **Purpose**: Prevents accidental overwriting without awareness of current content

### Overwrite Behavior
- **Complete Replacement**: Entire file content is replaced
- **No Merging**: Previous content is completely lost
- **Atomic Operation**: File is written atomically (all-or-nothing)
- **Backup Responsibility**: No automatic backup - use version control

## Usage Patterns

### New File Creation
```json
{
  "file_path": "/project/src/new-component.js",
  "content": "import React from 'react';\n\nconst NewComponent = () => {\n  return (\n    <div>\n      <h1>New Component</h1>\n    </div>\n  );\n};\n\nexport default NewComponent;"
}
```

### Configuration File Creation
```json
{
  "file_path": "/project/.env.example",
  "content": "# Environment Variables Example\nAPI_URL=https://api.example.com\nDATABASE_URL=postgresql://localhost:5432/myapp\nSECRET_KEY=your-secret-key-here\nDEBUG=false"
}
```

### Documentation Generation
```json
{
  "file_path": "/project/docs/API.md",
  "content": "# API Documentation\n\n## Authentication\n\nAll API requests require authentication.\n\n### POST /auth/login\n\nAuthenticate user and receive JWT token.\n\n**Request Body:**\n```json\n{\n  \"email\": \"user@example.com\",\n  \"password\": \"password123\"\n}\n```"
}
```

### Test File Creation
```json
{
  "file_path": "/project/tests/utils.test.js",
  "content": "const { validateEmail, formatDate } = require('../src/utils');\n\ndescribe('Utils', () => {\n  describe('validateEmail', () => {\n    test('validates correct email', () => {\n      expect(validateEmail('test@example.com')).toBe(true);\n    });\n\n    test('rejects invalid email', () => {\n      expect(validateEmail('invalid-email')).toBe(false);\n    });\n  });\n});"
}
```

### Script Generation
```json
{
  "file_path": "/project/scripts/deploy.sh",
  "content": "#!/bin/bash\n\nset -e\n\necho \"Building application...\"\nnpm run build\n\necho \"Running tests...\"\nnpm test\n\necho \"Deploying to production...\"\nrsync -avz dist/ user@server:/var/www/app/\n\necho \"Deployment complete!\""
}
```

## File Type Support

### Source Code Files
```javascript
// JavaScript/TypeScript
{
  "file_path": "/src/component.tsx",
  "content": "interface Props {\n  title: string;\n}\n\nconst Component: React.FC<Props> = ({ title }) => {\n  return <h1>{title}</h1>;\n};"
}

// Python
{
  "file_path": "/src/utils.py", 
  "content": "def calculate_total(items):\n    \"\"\"Calculate total price of items.\"\"\"\n    return sum(item['price'] for item in items)"
}
```

### Configuration Files
```json
// JSON Configuration
{
  "file_path": "/config/app.json",
  "content": "{\n  \"name\": \"MyApp\",\n  \"version\": \"1.0.0\",\n  \"database\": {\n    \"host\": \"localhost\",\n    \"port\": 5432\n  }\n}"
}

// YAML Configuration  
{
  "file_path": "/config/docker-compose.yml",
  "content": "version: '3.8'\nservices:\n  app:\n    build: .\n    ports:\n      - \"3000:3000\"\n  db:\n    image: postgres:13\n    environment:\n      POSTGRES_DB: myapp"
}
```

### Documentation Files
```markdown
// Markdown Documentation
{
  "file_path": "/docs/README.md",
  "content": "# Project Name\n\n## Description\n\nThis project provides...\n\n## Installation\n\n```bash\nnpm install\n```\n\n## Usage\n\n```javascript\nconst app = require('./app');\napp.start();\n```"
}
```

## Content Formatting

### Proper Newline Handling
```json
{
  "content": "Line 1\nLine 2\nLine 3"  // ✅ Correct
}

// Avoid escaped newlines unless specifically needed
{
  "content": "Line 1\\nLine 2\\nLine 3"  // ❌ Usually incorrect
}
```

### Indentation Preservation
```json
{
  "content": "function example() {\n    if (condition) {\n        return true;\n    }\n    return false;\n}"
}
```

### Special Characters
```json
{
  "content": "const regex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\nconst quote = \"He said, \\\"Hello, world!\\\"\";"
}
```

## Error Handling

### Common Errors

#### File Must Be Read First
```json
{
  "error": "File must be read before writing",
  "details": "Use Read tool on existing file before overwriting"
}
```

#### Directory Not Found
```json
{
  "error": "Parent directory does not exist: /nonexistent/path/",
  "suggestion": "Create parent directory first or use existing path"
}
```

#### Permission Denied
```json
{
  "error": "Permission denied: /protected/file.txt",
  "details": "Insufficient write permissions for target location"
}
```

### Recovery Strategies
1. **Read Existing**: Use Read tool before overwriting existing files
2. **Directory Creation**: Use Bash tool to create parent directories
3. **Permission Check**: Verify write permissions with LS tool
4. **Path Validation**: Ensure target path is correct and accessible

## Best Practices

### File Creation Guidelines
1. **Prefer Editing**: Use Edit/MultiEdit for existing files when possible
2. **New Files Only**: Use Write primarily for creating new files
3. **Version Control**: Ensure important files are under version control
4. **Backup Strategy**: Have backups for critical file overwrites

### Content Preparation
```json
// ✅ Good - Well-formatted, complete content
{
  "file_path": "/src/component.js",
  "content": "// Component header comment\nimport React from 'react';\n\nconst Component = () => {\n  return <div>Content</div>;\n};\n\nexport default Component;\n"
}

// ❌ Poor - Incomplete or malformed content
{
  "file_path": "/src/component.js", 
  "content": "import React from 'react';\nconst Component = () => {\nreturn <div>Content</div>"  // Missing closing brace
}
```

### Safety Practices
```json
// ✅ Safe - Creating new file
{
  "file_path": "/project/new-feature.js",
  "content": "// New feature implementation..."
}

// ⚠️ Risky - Overwriting existing file without read
// This would fail due to read-before-write policy
{
  "file_path": "/project/existing-file.js",
  "content": "// Replacement content..."
}
```

## Integration Patterns

### New Feature Workflow
```
1. Write(file_path="/src/new-feature.js", content="...")
2. Write(file_path="/tests/new-feature.test.js", content="...")
3. Bash(command="npm test new-feature.test.js")
4. Edit(file_path="/src/index.js", old_string="...", new_string="...")  // Add import
```

### Project Initialization
```
1. Write(file_path="/project/package.json", content="...")
2. Write(file_path="/project/README.md", content="...")
3. Write(file_path="/project/.gitignore", content="...")
4. Write(file_path="/project/src/index.js", content="...")
```

### Configuration Setup
```
1. Write(file_path="/config/development.json", content="...")
2. Write(file_path="/config/production.json", content="...")
3. Write(file_path="/config/test.json", content="...")
4. Bash(command="npm run validate-config")
```

## Performance Considerations

### File Size Guidelines
- **Small Files** (<10KB): Optimal performance
- **Medium Files** (10KB-100KB): Good performance
- **Large Files** (100KB-1MB): Consider if Write is appropriate
- **Very Large Files** (>1MB): Evaluate alternatives like streaming

### Memory Usage
- **Content Loading**: Entire content loaded into memory
- **Processing**: File written atomically after validation
- **Optimization**: For large content, consider breaking into smaller files

## Security Considerations

### File Path Safety
```json
// ✅ Safe - Project directory
{
  "file_path": "/home/user/projects/myapp/src/component.js"
}

// ⚠️ Risky - System directories
{
  "file_path": "/etc/hosts"  // Avoid system file modification
}

// ⚠️ Risky - Outside project scope
{
  "file_path": "/other/user/files/data.txt"  // Avoid accessing other user files
}
```

### Content Validation
- **No Secrets**: Avoid writing sensitive information to files
- **Input Sanitization**: Validate content doesn't contain malicious code
- **File Extensions**: Use appropriate extensions for content type
- **Permissions**: Consider file permissions after creation

## Tool Comparison

### Write vs Edit/MultiEdit
- **Write**: Complete file replacement, new file creation
- **Edit/MultiEdit**: Incremental changes, preserves most content

### Write vs NotebookEdit
- **Write**: Text files, complete replacement
- **NotebookEdit**: Jupyter notebooks, cell-level operations

### Write vs Bash
- **Write**: Controlled file creation with validation
- **Bash**: System commands, shell redirection (less safe)

## Advanced Usage

### Template Processing
```json
{
  "file_path": "/generated/component.js",
  "content": "import React from 'react';\n\nconst ${COMPONENT_NAME} = (${PROPS}) => {\n  return (\n    <div className=\"${CSS_CLASS}\">\n      ${CONTENT}\n    </div>\n  );\n};\n\nexport default ${COMPONENT_NAME};"
}
```

### Batch File Generation
```json
// Generate multiple related files
[
  {
    "file_path": "/api/routes/users.js",
    "content": "// User routes implementation..."
  },
  {
    "file_path": "/api/controllers/users.js", 
    "content": "// User controller implementation..."
  },
  {
    "file_path": "/tests/api/users.test.js",
    "content": "// User API tests..."
  }
]
```

### Dynamic Content Generation
```json
{
  "file_path": "/build/version.js",
  "content": "export const BUILD_VERSION = '1.2.3';\nexport const BUILD_DATE = '2024-01-15T10:30:00Z';\nexport const BUILD_HASH = 'abc123def456';"
}
```