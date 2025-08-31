import os
from tool_websearch import websearch

def test_websearch():
    """Test the websearch function with various scenarios."""
    
    # Set region to US for testing (since websearch is only available in US)
    original_region = os.environ.get('CLAUDE_REGION')
    os.environ['CLAUDE_REGION'] = 'US'
    
    try:
        # Test 1: Basic search
        result = websearch("Python programming")
        assert "results" in result
        assert "search_metadata" in result
        assert result["search_metadata"]["query"] == "Python programming"
        assert len(result["results"]) > 0
        print("✓ Basic search test passed")
        
        # Test 2: Search with allowed domains
        result = websearch("React tutorial", allowed_domains=["docs.com", "tutorial-site.com"])
        assert len(result["results"]) <= 2  # Should filter out example.com
        domains = [r["domain"] for r in result["results"]]
        assert all(domain in ["docs.com", "tutorial-site.com"] for domain in domains)
        print("✓ Allowed domains filtering test passed")
        
        # Test 3: Search with blocked domains  
        result = websearch("JavaScript guide", blocked_domains=["example.com"])
        domains = [r["domain"] for r in result["results"]]
        assert "example.com" not in domains
        assert "example.com" in result["search_metadata"]["filtered_domains"]
        print("✓ Blocked domains filtering test passed")
        
        # Test 4: Search with both allowed and blocked domains (non-conflicting)
        result = websearch("database tutorial", 
                          allowed_domains=["docs.com", "tutorial-site.com"],
                          blocked_domains=["example.com"])
        domains = [r["domain"] for r in result["results"]]
        assert "example.com" not in domains
        assert all(domain in ["docs.com", "tutorial-site.com"] for domain in domains)
        print("✓ Mixed domain filtering test passed")
        
        # Test 5: Empty query validation
        try:
            websearch("")
            assert False, "Should have raised ValueError for empty query"
        except ValueError as e:
            assert "non-empty string" in str(e)
            print("✓ Empty query validation test passed")
        
        # Test 6: Short query validation
        try:
            websearch("a")
            assert False, "Should have raised ValueError for short query"
        except ValueError as e:
            assert "at least 2 characters" in str(e)
            print("✓ Short query validation test passed")
        
        # Test 7: Non-string query validation
        try:
            websearch(123)
            assert False, "Should have raised ValueError for non-string query"
        except ValueError as e:
            assert "non-empty string" in str(e)
            print("✓ Non-string query validation test passed")
        
        # Test 8: Invalid allowed_domains type
        try:
            websearch("test query", allowed_domains="not-a-list")
            assert False, "Should have raised ValueError for non-list allowed_domains"
        except ValueError as e:
            assert "must be a list" in str(e)
            print("✓ Invalid allowed_domains type test passed")
        
        # Test 9: Invalid blocked_domains type
        try:
            websearch("test query", blocked_domains="not-a-list")
            assert False, "Should have raised ValueError for non-list blocked_domains" 
        except ValueError as e:
            assert "must be a list" in str(e)
            print("✓ Invalid blocked_domains type test passed")
        
        # Test 10: Conflicting domain filters
        try:
            websearch("test query", 
                     allowed_domains=["example.com", "docs.com"],
                     blocked_domains=["example.com", "other.com"])
            assert False, "Should have raised ValueError for conflicting domains"
        except ValueError as e:
            assert "cannot be both allowed and blocked" in str(e)
            print("✓ Conflicting domain filters test passed")
        
        # Test 11: Geographic restriction
        os.environ['CLAUDE_REGION'] = 'EU'
        try:
            websearch("test query")
            assert False, "Should have raised RuntimeError for non-US region"
        except RuntimeError as e:
            assert "not available in your region" in str(e)
            print("✓ Geographic restriction test passed")
        finally:
            os.environ['CLAUDE_REGION'] = 'US'
        
        # Test 12: Result structure validation
        result = websearch("API documentation")
        assert isinstance(result, dict)
        assert "results" in result
        assert "search_metadata" in result
        
        # Check result structure
        for search_result in result["results"]:
            required_fields = ["title", "url", "snippet", "domain", "relevance_score", "publish_date", "source_type"]
            for field in required_fields:
                assert field in search_result, f"Missing field: {field}"
        
        # Check metadata structure
        metadata = result["search_metadata"]
        required_metadata = ["query", "total_results", "search_time", "filtered_domains"]
        for field in required_metadata:
            assert field in metadata, f"Missing metadata field: {field}"
        
        print("✓ Result structure validation test passed")
        
        # Test 13: Empty domain lists (should work fine)
        result = websearch("test query", allowed_domains=[], blocked_domains=[])
        assert "results" in result
        print("✓ Empty domain lists test passed")
        
        print("\nAll tests passed! ✅")
        
    finally:
        # Restore original region
        if original_region is not None:
            os.environ['CLAUDE_REGION'] = original_region
        elif 'CLAUDE_REGION' in os.environ:
            del os.environ['CLAUDE_REGION']

if __name__ == "__main__":
    test_websearch()