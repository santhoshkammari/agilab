# WebFetch Tool - Comprehensive Documentation

## Overview
The WebFetch tool fetches content from URLs and processes it using AI analysis. It converts HTML to markdown, processes the content with a specified prompt, and returns AI-generated insights about the web content.

## Function Signature
```json
{
  "name": "WebFetch",
  "parameters": {
    "url": "string (required)",
    "prompt": "string (required)"
  }
}
```

## Parameters

### url (required)
- **Type**: string (URI format)
- **Purpose**: URL to fetch content from
- **Validation**: Must be fully-formed valid URL
- **Protocol**: HTTP URLs automatically upgraded to HTTPS
- **Format**: `https://example.com/path`

### prompt (required)
- **Type**: string
- **Purpose**: Instructions for AI processing of the fetched content
- **Content**: Describes what information to extract or analyze
- **Processing**: Prompt is applied to HTML-to-markdown converted content
- **Response**: AI model returns analysis based on this prompt

## Content Processing Pipeline

### 1. URL Fetching
```
Input URL → HTTP/HTTPS Request → Raw HTML Content
```
- **Protocol Upgrade**: HTTP automatically converted to HTTPS
- **Headers**: Standard browser-like headers sent
- **Timeout**: Reasonable timeout for web requests
- **Error Handling**: Network errors caught and reported

### 2. Content Conversion
```
Raw HTML → Markdown Conversion → Structured Text
```
- **HTML Parsing**: Extracts meaningful content from HTML
- **Markdown Format**: Converts to clean, readable markdown
- **Structure Preservation**: Maintains headings, lists, links
- **Content Cleaning**: Removes ads, navigation, footers

### 3. AI Processing
```
Markdown Content + Prompt → AI Model → Structured Response
```
- **Model**: Small, fast AI model optimized for content analysis
- **Context**: Full page content provided to model
- **Prompt Application**: User prompt guides analysis focus
- **Response Generation**: Structured, relevant response returned

## Usage Patterns

### Documentation Analysis
```json
{
  "url": "https://docs.python.org/3/library/requests.html",
  "prompt": "Extract the main features and basic usage examples of the requests library"
}
```

### API Documentation Review
```json
{
  "url": "https://api.example.com/docs",
  "prompt": "Summarize the available endpoints, authentication methods, and rate limits"
}
```

### News Article Summarization
```json
{
  "url": "https://techblog.example.com/article",
  "prompt": "Provide a concise summary of the main points and key takeaways"
}
```

### Tutorial Extraction
```json
{
  "url": "https://tutorial.example.com/getting-started",
  "prompt": "Extract the step-by-step instructions and list any prerequisites"
}
```

### Technology Research
```json
{
  "url": "https://github.com/project/awesome-project",
  "prompt": "Analyze the project features, installation requirements, and usage examples"
}
```

## Redirect Handling

### Redirect Detection
When a URL redirects to a different host, the tool provides:
```json
{
  "redirect_detected": true,
  "original_url": "https://short.ly/abc123",
  "redirect_url": "https://actualdestination.com/full-article",
  "message": "URL redirected to different host. Use redirect_url for content."
}
```

### Follow-up Request
```json
// Required follow-up request with redirect URL
{
  "url": "https://actualdestination.com/full-article",
  "prompt": "Extract the main content and key points from this article"
}
```

### Cross-domain Redirects
- **Detection**: Automatic detection of host changes
- **Security**: Prevents automatic following of cross-domain redirects
- **User Control**: Requires explicit user confirmation via new request
- **Transparency**: Clear communication about redirect destination

## Caching System

### Cache Behavior
- **Duration**: 15-minute self-cleaning cache
- **Key**: URL-based cache keys
- **Benefit**: Faster responses for repeated requests
- **Automatic**: No user intervention required

### Cache Optimization
```json
// First request (fetches from web)
{
  "url": "https://docs.example.com/api",
  "prompt": "Extract API endpoints"
}

// Second request within 15 minutes (uses cache)
{
  "url": "https://docs.example.com/api", 
  "prompt": "Find authentication examples"
}
```

### Cache Invalidation
- **Time-based**: Automatic expiration after 15 minutes
- **Content**: Fresh content fetched after expiration
- **Reliability**: Ensures reasonably current information

## Response Format

### Successful Response
```json
{
  "success": true,
  "url": "https://example.com/article",
  "content_summary": "AI-processed response based on prompt",
  "metadata": {
    "title": "Page Title",
    "fetch_time": "2024-01-15T10:30:00Z",
    "content_length": 5420,
    "from_cache": false
  }
}
```

### Error Response
```json
{
  "error": true,
  "message": "Failed to fetch URL: Connection timeout",
  "url": "https://unreachable.example.com",
  "error_code": "TIMEOUT"
}
```

### Redirect Response
```json
{
  "redirect": true,
  "original_url": "https://short.url/abc",
  "redirect_url": "https://destination.com/article",
  "message": "Please make new request with redirect_url"
}
```

## Prompt Engineering

### Effective Prompts
```json
// ✅ Specific, actionable prompts
{
  "prompt": "Extract the installation steps, system requirements, and basic configuration options"
}

{
  "prompt": "Summarize the main features, pricing tiers, and API limitations mentioned on this page"
}

{
  "prompt": "List all the code examples shown and explain what each one demonstrates"
}
```

### Less Effective Prompts
```json
// ❌ Too vague
{
  "prompt": "Tell me about this page"
}

// ❌ Too broad
{
  "prompt": "Extract everything important"
}
```

### Structured Prompts
```json
{
  "prompt": "Analyze this documentation and provide: 1) Main purpose of the tool, 2) Installation instructions, 3) Basic usage examples, 4) Key features and limitations"
}
```

## Content Types

### Documentation Sites
- **Technical Docs**: API references, user guides, tutorials
- **Project READMEs**: GitHub project documentation
- **Reference Materials**: Language references, framework docs
- **How-to Guides**: Step-by-step instructions

### News and Articles
- **Tech News**: Industry updates, product announcements
- **Blog Posts**: Technical articles, opinion pieces
- **Research Papers**: Academic or industry research
- **Case Studies**: Implementation examples, lessons learned

### Project Pages
- **GitHub Repositories**: Project information, README content
- **Product Pages**: Feature descriptions, specifications
- **Landing Pages**: Marketing content, feature overviews
- **Release Notes**: Version updates, changelog information

## Error Handling

### Network Errors
```json
{
  "error": "Network timeout",
  "details": "Failed to connect to server within timeout period",
  "retry_suggestion": "Check URL and try again"
}
```

### Invalid URLs
```json
{
  "error": "Invalid URL format",
  "details": "URL must be fully-formed (include protocol)",
  "example": "https://example.com/path"
}
```

### Access Restrictions
```json
{
  "error": "Access denied",
  "details": "Server returned 403 Forbidden",
  "suggestion": "URL may require authentication or be restricted"
}
```

### Content Processing Errors
```json
{
  "error": "Content processing failed",
  "details": "Unable to extract meaningful content from HTML",
  "suggestion": "URL may point to non-text content or require JavaScript"
}
```

## Security Considerations

### URL Validation
- **Protocol Requirements**: Only HTTP/HTTPS supported
- **Malicious URL Detection**: Basic validation for obviously malicious URLs
- **User Responsibility**: Users should verify URL safety
- **No Execution**: Tool only reads content, doesn't execute scripts

### Content Safety
- **Read-only**: Tool doesn't modify any external content
- **No Authentication**: Doesn't handle login credentials
- **Public Content**: Only accesses publicly available content
- **Privacy**: No personal information transmitted to external sites

### Data Handling
- **Temporary Processing**: Content processed temporarily for analysis
- **No Persistence**: Original HTML not stored long-term
- **AI Processing**: Content sent to AI model for analysis
- **Cache Limitation**: 15-minute cache for performance only

## Performance Considerations

### Response Time
- **Network Speed**: Dependent on target site response time
- **Content Size**: Larger pages take longer to process
- **AI Processing**: Additional time for content analysis
- **Cache Benefit**: Significant speedup for cached content

### Content Limitations
- **Size Limits**: Very large pages may be truncated
- **Complex Sites**: JavaScript-heavy sites may not render fully
- **Dynamic Content**: Real-time data may not be captured
- **Media Content**: Images/videos not processed, only referenced

### Optimization Strategies
```json
// Use specific prompts to reduce processing time
{
  "prompt": "Extract only the installation section from this documentation"
}

// Leverage caching for related requests
{
  "url": "https://docs.example.com",
  "prompt": "Get overview"
}
// Follow with:
{
  "url": "https://docs.example.com",  // Same URL uses cache
  "prompt": "Find troubleshooting section"
}
```

## Integration Patterns

### Research Workflow
```
1. WebFetch(url="...", prompt="Get overview")
2. WebFetch(url="...", prompt="Find technical details")
3. Document findings in local files
4. Use information for implementation
```

### Documentation Review
```
1. WebFetch(url="API_docs", prompt="Extract endpoints")
2. WebFetch(url="GitHub_repo", prompt="Get setup instructions")
3. Write(file_path="implementation-notes.md", content="...")
```

### Competitive Analysis
```
1. WebFetch(url="competitor1", prompt="Analyze features")
2. WebFetch(url="competitor2", prompt="Analyze features")
3. Compare findings and document insights
```

## Best Practices

### URL Selection
1. **Direct Links**: Use direct links to specific content
2. **Avoid Redirects**: Use final URLs when known
3. **Public Content**: Ensure URLs are publicly accessible
4. **Stable URLs**: Prefer permanent URLs over temporary ones

### Prompt Design
1. **Be Specific**: Clear, focused prompts yield better results
2. **Structured Requests**: Use numbered lists for complex analysis
3. **Context Aware**: Consider what type of content the URL contains
4. **Actionable**: Ask for specific, actionable information

### Error Recovery
1. **URL Verification**: Check URLs are correct and accessible
2. **Retry Strategy**: Wait and retry for temporary network issues
3. **Alternative Sources**: Have backup URLs for critical information
4. **Prompt Adjustment**: Modify prompts if initial results are unclear

## MCP Tool Priority

### Important Note
```
IMPORTANT: If an MCP-provided web fetch tool is available (tools starting with "mcp__"), 
prefer using that tool instead of this one, as it may have fewer restrictions.
```

### Tool Selection Priority
1. **MCP Tools**: Use `mcp__*` web tools if available
2. **WebFetch**: Use this tool if no MCP alternatives
3. **Capability Check**: Verify MCP tool capabilities first
4. **Feature Comparison**: Compare features and limitations