# Bash Tool - Comprehensive Documentation

## Overview
The Bash tool executes bash commands in a persistent shell session with security measures, timeout controls, and proper handling of command output. It maintains session state across multiple invocations.

## Function Signature
```json
{
  "name": "Bash",
  "parameters": {
    "command": "string (required)",
    "description": "string (optional)",
    "timeout": "number (optional, max 600000)"
  }
}
```

## Parameters

### command (required)
- **Type**: string
- **Purpose**: The bash command to execute
- **Constraints**: 
  - Must be valid bash syntax
  - Paths with spaces must be quoted with double quotes
  - No interactive commands (avoid `-i` flags)
- **Examples**:
  - `"ls -la"`
  - `"cd \"/path/with spaces\""`
  - `"python script.py"`

### description (optional)
- **Type**: string
- **Purpose**: 5-10 word description of what the command does
- **Format**: Clear, concise action description
- **Examples**:
  - `"Lists files in current directory"`
  - `"Shows working tree status"`
  - `"Installs package dependencies"`

### timeout (optional)
- **Type**: number
- **Unit**: milliseconds
- **Default**: 120000ms (2 minutes)
- **Maximum**: 600000ms (10 minutes)
- **Purpose**: Prevents hanging commands

## Session Management

### Persistent Shell
- **State Preservation**: Working directory, environment variables, shell state persist
- **Session Continuity**: Commands executed in same shell instance
- **Memory**: Previous command outputs and side effects remain

### Working Directory Management
- **Best Practice**: Use absolute paths instead of `cd`
- **Acceptable**: `cd` when user explicitly requests it
- **Recommendation**: Maintain current directory throughout session

```bash
# Preferred
pytest /foo/bar/tests

# Less preferred
cd /foo/bar && pytest tests
```

## Security and Safety

### Command Restrictions
- **Prohibited**: Search commands (`find`, `grep`) - use Grep/Glob tools instead
- **Prohibited**: File reading (`cat`, `head`, `tail`, `ls`) - use Read/LS tools instead
- **Required**: Use `rg` (ripgrep) if grep functionality needed
- **Interactive**: Avoid commands requiring user input

### Path Handling
- **Spaces**: Always quote paths containing spaces
- **Validation**: Verify parent directories exist before creating new ones
- **Examples**:
  ```bash
  # Correct
  cd "/Users/name/My Documents"
  python "/path/with spaces/script.py"
  
  # Incorrect - will fail
  cd /Users/name/My Documents
  python /path/with spaces/script.py
  ```

### Pre-execution Checks
1. **Directory Verification**: Use LS tool to verify parent directories exist
2. **Path Validation**: Ensure target locations are correct
3. **Command Review**: Check for proper quoting and syntax

## Output Handling

### Response Format
- **Success**: Command output to stdout
- **Errors**: stderr content included
- **Exit Codes**: Implicit in success/failure indication
- **Truncation**: Output exceeds 30,000 characters gets truncated

### Output Processing
- **Encoding**: UTF-8 text processing
- **Binary**: Binary output may not display correctly
- **Large Files**: Use Read tool for large file content instead

## Command Composition

### Multiple Commands
- **Separator Options**:
  - `;` - Execute sequentially regardless of success
  - `&&` - Execute only if previous succeeds
  - `||` - Execute only if previous fails

- **Prohibited**: Newlines for command separation
- **Allowed**: Newlines within quoted strings

```bash
# Good
git add . && git commit -m "Update files" && git push

# Good
echo "Line 1
Line 2" > multiline.txt

# Bad - don't use newlines to separate commands
git add .
git commit -m "Update"
```

## Git Operations

### Status and Information
```bash
git status                    # Show working tree status
git diff                      # Show unstaged changes
git diff --staged             # Show staged changes
git log --oneline -10         # Recent commit history
```

### Staging and Committing
```bash
git add .                     # Stage all changes
git add specific-file.js      # Stage specific file
git commit -m "$(cat <<'EOF'
Commit message here.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### Commit Message Format
- **HEREDOC Required**: Always use HEREDOC for commit messages
- **Footer Required**: Include generation attribution
- **Style**: Follow repository's existing commit message patterns

### Pull Request Creation
```bash
# Check current state
git status && git log --oneline -5 && git diff main...HEAD

# Push and create PR
git push -u origin feature-branch
gh pr create --title "PR Title" --body "$(cat <<'EOF'
## Summary
- Bullet point 1
- Bullet point 2

## Test plan
- [ ] Test item 1
- [ ] Test item 2

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

## Package Management

### Node.js
```bash
npm install                   # Install dependencies
npm run build                 # Build project
npm test                      # Run tests
npm run lint                  # Run linting
```

### Python
```bash
pip install -r requirements.txt
python -m pytest
python setup.py build
```

### System Packages
```bash
sudo apt update && sudo apt install package-name
brew install package-name
```

## File Operations

### Directory Operations
```bash
mkdir -p /path/to/nested/dirs  # Create nested directories
rmdir empty-directory          # Remove empty directory
rm -rf directory-with-content  # Remove directory and contents
```

### File Manipulation
```bash
touch new-file.txt            # Create empty file
cp source.txt dest.txt        # Copy file
mv old-name.txt new-name.txt  # Move/rename file
chmod +x script.sh            # Make executable
```

## Process Management

### Background Processes
```bash
nohup long-running-command &  # Run in background
jobs                          # List background jobs
kill %1                       # Kill job by number
pkill -f process-name         # Kill by process name
```

### System Information
```bash
ps aux | head -20             # Show running processes
df -h                         # Disk usage
free -h                       # Memory usage
top -n 1                      # System resources snapshot
```

## Error Handling

### Common Error Types
1. **Command Not Found**: Command doesn't exist or not in PATH
2. **Permission Denied**: Insufficient permissions for operation
3. **File Not Found**: Target file/directory doesn't exist
4. **Timeout**: Command exceeded timeout limit
5. **Syntax Error**: Invalid bash syntax

### Debugging Strategies
```bash
# Check if command exists
which command-name
type command-name

# Check permissions
ls -la file-or-directory

# Test with verbose output
command -v  # or --verbose flag
```

## Performance Considerations

### Timeout Management
- **Default**: 2 minutes for most operations
- **Extended**: Use longer timeouts for builds, installs
- **Monitoring**: Commands that may hang should have appropriate timeouts

### Resource Usage
- **Memory**: Large output gets truncated
- **CPU**: Long-running processes may impact system
- **Disk**: Be mindful of space usage with file operations

## Integration Patterns

### With Other Tools
- **Pre-checks**: Use LS to verify directories before bash operations
- **Post-verification**: Use Read to confirm file changes
- **Error Recovery**: Use Grep to search for error patterns in logs

### Workflow Examples
1. **Build Process**:
   ```bash
   npm install && npm run build && npm test
   ```

2. **Git Workflow**:
   ```bash
   git status && git add . && git commit -m "message" && git push
   ```

3. **System Setup**:
   ```bash
   mkdir -p project/src && cd project && npm init -y
   ```

## Best Practices

### Command Construction
1. **Atomic Operations**: Prefer single-purpose commands
2. **Error Handling**: Use `&&` to chain dependent operations
3. **Path Safety**: Always quote paths with spaces
4. **Tool Selection**: Use specialized tools (Grep, Read) over bash equivalents

### Session Management
1. **State Awareness**: Remember that shell state persists
2. **Clean Operations**: Don't leave processes running unless intended
3. **Directory Hygiene**: Return to appropriate working directory after operations

### Security Practices
1. **Input Validation**: Validate file paths before operations
2. **Permission Awareness**: Use appropriate permissions for file operations
3. **Credential Safety**: Never expose secrets in command output
4. **System Impact**: Be mindful of system resource usage