def websearch(query, allowed_domains=None, blocked_domains=None):
    """
    Perform a web search and return formatted search results.
    
    Args:
        query (str): Search terms to find relevant web content (minimum 2 characters)
        allowed_domains (list, optional): List of domains to whitelist for results
        blocked_domains (list, optional): List of domains to blacklist from results
        
    Returns:
        dict: Search results with metadata in the following format:
        {
            "results": [
                {
                    "title": "Page Title - Site Name",
                    "url": "https://example.com/page",
                    "snippet": "Brief description with relevant keywords...",
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
        
    Raises:
        ValueError: If query is too short or invalid parameters
        RuntimeError: If web search is not available in current region
        ConnectionError: If search service is unavailable
    """
    # Validate query
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    if len(query.strip()) < 2:
        raise ValueError("Query must be at least 2 characters long")
    
    # Validate domain lists
    if allowed_domains is not None:
        if not isinstance(allowed_domains, list):
            raise ValueError("allowed_domains must be a list")
        if not all(isinstance(domain, str) for domain in allowed_domains):
            raise ValueError("All domains in allowed_domains must be strings")
    
    if blocked_domains is not None:
        if not isinstance(blocked_domains, list):
            raise ValueError("blocked_domains must be a list")
        if not all(isinstance(domain, str) for domain in blocked_domains):
            raise ValueError("All domains in blocked_domains must be strings")
    
    # Check for conflicting domain filters
    if allowed_domains and blocked_domains:
        overlap = set(allowed_domains) & set(blocked_domains)
        if overlap:
            raise ValueError(f"Domains cannot be both allowed and blocked: {list(overlap)}")
    
    # Mock implementation for testing - in real implementation this would call actual search API
    # This simulates the actual behavior described in the documentation
    
    # Simulate geographic restriction
    import os
    region = os.environ.get('CLAUDE_REGION', 'US')
    if region != 'US':
        raise RuntimeError("Web search not available in your region. Available regions: United States")
    
    # Simulate search processing
    import time
    search_start = time.time()
    
    # Mock search results based on query
    mock_results = [
        {
            "title": f"Search Result for '{query}' - Example.com",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}", 
            "snippet": f"This is a mock search result for the query '{query}'. It contains relevant information and highlights the search terms.",
            "domain": "example.com",
            "relevance_score": 0.95,
            "publish_date": "2025-08-01",
            "source_type": "article"
        },
        {
            "title": f"Documentation for '{query}' - Docs.com",
            "url": f"https://docs.com/topics/{query.replace(' ', '-').lower()}",
            "snippet": f"Official documentation covering '{query}' with comprehensive examples and best practices for implementation.",
            "domain": "docs.com", 
            "relevance_score": 0.87,
            "publish_date": "2025-07-15",
            "source_type": "documentation"
        },
        {
            "title": f"Tutorial: {query} - Tutorial Site",
            "url": f"https://tutorial-site.com/learn/{query.replace(' ', '-')}",
            "snippet": f"Step-by-step tutorial on {query} with practical examples and code samples for beginners and experts.",
            "domain": "tutorial-site.com",
            "relevance_score": 0.82,
            "publish_date": "2025-06-20", 
            "source_type": "tutorial"
        }
    ]
    
    # Apply domain filtering
    filtered_results = []
    filtered_domains = []
    
    for result in mock_results:
        domain = result["domain"]
        
        # Check allowed domains filter
        if allowed_domains is not None:
            if domain not in allowed_domains:
                filtered_domains.append(domain)
                continue
        
        # Check blocked domains filter
        if blocked_domains is not None:
            if domain in blocked_domains:
                filtered_domains.append(domain) 
                continue
                
        filtered_results.append(result)
    
    search_time = time.time() - search_start
    
    # Return formatted results
    return {
        "results": filtered_results,
        "search_metadata": {
            "query": query,
            "total_results": len(filtered_results),
            "search_time": round(search_time, 3),
            "filtered_domains": list(set(filtered_domains)) if filtered_domains else []
        }
    }