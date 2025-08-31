# WebSearch Tool - Comprehensive Documentation

## Overview
The WebSearch tool performs web searches and returns formatted search result blocks. It provides access to current information beyond the AI's knowledge cutoff and supports domain filtering for targeted searches.

## Function Signature
```json
{
  "name": "WebSearch",
  "parameters": {
    "query": "string (required, min 2 characters)",
    "allowed_domains": "array of strings (optional)",
    "blocked_domains": "array of strings (optional)"
  }
}
```

## Parameters

### query (required)
- **Type**: string
- **Minimum Length**: 2 characters
- **Purpose**: Search terms to find relevant web content
- **Format**: Natural language search query
- **Best Practices**: Use specific, descriptive terms

### allowed_domains (optional)
- **Type**: array of strings
- **Purpose**: Whitelist specific domains for search results
- **Format**: Domain names without protocol (e.g., "example.com")
- **Behavior**: Only results from these domains will be included
- **Use Case**: Focused research from trusted sources

### blocked_domains (optional)
- **Type**: array of strings
- **Purpose**: Blacklist specific domains from search results
- **Format**: Domain names without protocol (e.g., "spam.com")
- **Behavior**: Results from these domains will be excluded
- **Use Case**: Filtering out unreliable or unwanted sources

## Geographic Availability

### Regional Restrictions
- **Availability**: Web search only available in the US
- **Detection**: Service automatically detects user location
- **Limitation**: International users cannot access web search
- **Alternative**: Use WebFetch for known URLs outside US

## Date Awareness

### Current Information
- **Today's Date**: System aware of current date (2025-08-02 in environment)
- **Query Optimization**: Automatically accounts for current year in searches
- **Recent Content**: Prioritizes recent information when relevant
- **Example**: Searching for "latest Python features" will include 2025 content

### Temporal Queries
```json
// ✅ Good - Current year aware
{
  "query": "Python 3.12 new features 2025"
}

// ❌ Poor - Outdated year reference
{
  "query": "latest Python features 2024"  // When current year is 2025
}
```

## Search Result Format

### Result Block Structure
```json
{
  "results": [
    {
      "title": "Page Title - Site Name",
      "url": "https://example.com/page",
      "snippet": "Brief description of the page content with relevant keywords highlighted...",
      "domain": "example.com",
      "relevance_score": 0.95,
      "publish_date": "2025-01-15",
      "source_type": "article"
    }
  ],
  "search_metadata": {
    "query": "original search terms",
    "total_results": 847,
    "search_time": 0.23,
    "filtered_domains": ["domain1.com", "domain2.com"]
  }
}
```

### Result Properties
- **title**: Page title and site name
- **url**: Direct link to the content
- **snippet**: Relevant excerpt with search term context
- **domain**: Source domain for filtering reference
- **relevance_score**: Search engine relevance rating
- **publish_date**: Content publication date (when available)
- **source_type**: Content type (article, documentation, forum, etc.)

## Usage Patterns

### Technology Research
```json
{
  "query": "React 18 concurrent features best practices 2025",
  "allowed_domains": ["react.dev", "github.com", "stackoverflow.com"]
}
```

### Documentation Search
```json
{
  "query": "Python asyncio tutorial examples",
  "allowed_domains": ["python.org", "docs.python.org", "realpython.com"]
}
```

### Current Events
```json
{
  "query": "AI safety regulations 2025 latest developments"
}
```

### Problem Solving
```json
{
  "query": "debugging Node.js memory leaks production",
  "blocked_domains": ["unreliable-tutorials.com", "spam-blog.net"]
}
```

### Competitive Analysis
```json
{
  "query": "cloud storage providers comparison 2025 features pricing"
}
```

## Domain Filtering

### Allowed Domains Strategy
```json
{
  "query": "machine learning algorithms comparison",
  "allowed_domains": [
    "scikit-learn.org",
    "tensorflow.org", 
    "pytorch.org",
    "kaggle.com",
    "towards-datascience.medium.com"
  ]
}
```

### Blocked Domains Strategy
```json
{
  "query": "cybersecurity best practices",
  "blocked_domains": [
    "clickbait-tech.com",
    "fake-security-blog.net",
    "spam-content-farm.org"
  ]
}
```

### Mixed Filtering
```json
{
  "query": "database performance optimization",
  "allowed_domains": ["postgresql.org", "mysql.com", "mongodb.com"],
  "blocked_domains": ["outdated-db-blog.com"]
}
```

## Search Query Optimization

### Effective Search Terms
```json
// ✅ Specific and targeted
{
  "query": "Docker multi-stage builds optimization techniques 2025"
}

// ✅ Problem-focused
{
  "query": "React state management Redux vs Context API performance"
}

// ✅ Technology-specific
{
  "query": "TypeScript strict mode configuration best practices"
}
```

### Less Effective Queries
```json
// ❌ Too generic
{
  "query": "programming"
}

// ❌ Too broad
{
  "query": "web development"
}

// ❌ Outdated temporal reference
{
  "query": "JavaScript frameworks 2020"  // When seeking current info
}
```

### Query Enhancement Techniques
```json
// Add context and specificity
{
  "query": "GraphQL vs REST API performance comparison enterprise applications"
}

// Include error messages or specific problems
{
  "query": "CORS error XMLHttpRequest blocked different origin solutions"
}

// Specify technology versions
{
  "query": "Angular 17 standalone components migration guide"
}
```

## Information Currency

### Recent Developments
```json
{
  "query": "OpenAI GPT-4 API pricing changes 2025"
}
```

### Technology Updates
```json
{
  "query": "Kubernetes 1.29 new features deprecated APIs"
}
```

### Industry News
```json
{
  "query": "cloud computing trends enterprise adoption 2025"
}
```

### Breaking Changes
```json
{
  "query": "Node.js 21 breaking changes migration guide"
}
```

## Integration Patterns

### Research Workflow
```
1. WebSearch(query="initial research topic")
2. Review search results for relevant sources
3. WebFetch(url="specific_result_url", prompt="extract details")
4. Write(file_path="research-notes.md", content="compiled findings")
```

### Problem Solving Workflow
```
1. WebSearch(query="specific error message or problem")
2. Filter results for reliable sources
3. WebFetch(url="solution_url", prompt="extract step-by-step solution")
4. Implement solution in code
```

### Technology Evaluation
```
1. WebSearch(query="technology A vs technology B comparison")
2. WebSearch(query="technology A pros cons 2025", allowed_domains=["trusted-sources"])
3. Compile findings for decision making
```

## Performance Considerations

### Search Speed
- **Response Time**: Typically 200ms-2s depending on query complexity
- **Concurrent Searches**: Multiple searches can be performed in parallel
- **Result Processing**: Additional time for result formatting and filtering
- **Domain Filtering**: May reduce response time by limiting search scope

### Result Quality
- **Relevance Ranking**: Results ordered by search engine relevance
- **Freshness**: Recent content prioritized for time-sensitive queries
- **Authority**: Reputable domains typically rank higher
- **Filtering Impact**: Domain filters may affect result diversity

### Optimization Strategies
```json
// Parallel searches for comprehensive coverage
[
  {
    "query": "React performance optimization",
    "allowed_domains": ["react.dev", "web.dev"]
  },
  {
    "query": "React performance monitoring tools",
    "allowed_domains": ["github.com", "npm.js"]
  }
]
```

## Error Handling

### Common Errors

#### Query Too Short
```json
{
  "error": "Query must be at least 2 characters long",
  "provided_query": "a",
  "minimum_length": 2
}
```

#### Geographic Restriction
```json
{
  "error": "Web search not available in your region",
  "available_regions": ["United States"],
  "alternative": "Use WebFetch with known URLs"
}
```

#### No Results Found
```json
{
  "results": [],
  "message": "No results found for query",
  "suggestions": [
    "Try different search terms",
    "Remove domain restrictions",
    "Use broader search terms"
  ]
}
```

#### Domain Filter Conflict
```json
{
  "error": "No results after domain filtering",
  "query": "search terms",
  "allowed_domains": ["very-specific-domain.com"],
  "suggestion": "Broaden allowed domains or remove restrictions"
}
```

### Recovery Strategies
1. **Query Refinement**: Adjust search terms for better results
2. **Domain Adjustment**: Modify allowed/blocked domain lists
3. **Alternative Sources**: Use WebFetch for known reliable URLs
4. **Query Expansion**: Add synonyms or related terms

## Best Practices

### Query Construction
1. **Be Specific**: Use precise, descriptive terms
2. **Include Context**: Add relevant technical context
3. **Current Year**: Reference current year for latest information
4. **Problem Focus**: Include specific problems or error messages

### Domain Management
```json
// ✅ Strategic domain filtering
{
  "query": "API security best practices",
  "allowed_domains": [
    "owasp.org",
    "security.googleblog.com", 
    "auth0.com",
    "okta.com"
  ]
}

// ✅ Balanced approach - block known bad actors
{
  "query": "machine learning tutorials",
  "blocked_domains": ["content-farm.com", "plagiarized-content.net"]
}
```

### Information Validation
1. **Source Verification**: Check domain credibility
2. **Date Relevance**: Ensure information is current
3. **Cross-Reference**: Compare multiple sources
4. **Technical Accuracy**: Verify technical details with official docs

## Security Considerations

### Search Privacy
- **Query Logging**: Search queries may be logged by search provider
- **Result Filtering**: Personal information should not be included in queries
- **Domain Safety**: Verify domains before visiting result URLs
- **Content Validation**: Verify information from search results

### Information Security
- **Source Verification**: Not all search results are reliable
- **Link Safety**: Exercise caution when following result URLs
- **Content Accuracy**: Validate technical information from multiple sources
- **Credential Safety**: Never include credentials or secrets in search queries

## Advanced Features

### Targeted Research
```json
{
  "query": "GraphQL schema design patterns microservices",
  "allowed_domains": [
    "apollographql.com",
    "github.com",
    "hasura.io",
    "graphql.org"
  ]
}
```

### Competitive Intelligence
```json
{
  "query": "serverless computing AWS vs Azure vs GCP 2025 comparison"
}
```

### Trend Analysis
```json
{
  "query": "JavaScript framework popularity trends 2025 developer survey"
}
```

### Technical Documentation Discovery
```json
{
  "query": "Redis cluster configuration high availability",
  "allowed_domains": ["redis.io", "redis.com"]
}
```