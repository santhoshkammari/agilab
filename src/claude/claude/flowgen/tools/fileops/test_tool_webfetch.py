import time
from unittest.mock import patch, Mock
from tool_webfetch import web_fetch, clear_cache, _generate_cache_key, _get_from_cache, _store_in_cache


def test_web_fetch():
    """Test the web_fetch function with various scenarios."""
    
    # Clear cache before tests
    clear_cache()
    
    # Test 1: Input validation - empty URL
    try:
        web_fetch("", "test prompt")
        assert False, "Should have raised ValueError for empty URL"
    except ValueError as e:
        assert "url cannot be empty" in str(e)
        print("✓ Empty URL validation test passed")
    
    # Test 2: Input validation - empty prompt
    try:
        web_fetch("https://example.com", "")
        assert False, "Should have raised ValueError for empty prompt"
    except ValueError as e:
        assert "prompt cannot be empty" in str(e)
        print("✓ Empty prompt validation test passed")
    
    # Test 3: Input validation - invalid URL format
    try:
        web_fetch("not-a-url", "test prompt")
        assert False, "Should have raised ValueError for invalid URL"
    except ValueError as e:
        assert "must include protocol" in str(e)
        print("✓ Invalid URL format test passed")
    
    # Test 4: Input validation - non-string inputs
    try:
        web_fetch(123, "test prompt")
        assert False, "Should have raised TypeError for non-string URL"
    except TypeError as e:
        assert "url must be a string" in str(e)
        print("✓ Non-string URL validation test passed")
    
    try:
        web_fetch("https://example.com", 123)
        assert False, "Should have raised TypeError for non-string prompt"
    except TypeError as e:
        assert "prompt must be a string" in str(e)
        print("✓ Non-string prompt validation test passed")
    
    # Test 5: HTTP to HTTPS upgrade
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test Page</title></head><body><h1>Hello World</h1></body></html>"
        mock_get.return_value = mock_response
        
        result = web_fetch("http://example.com", "Extract the main heading")
        
        # Check that the request was made with HTTPS
        mock_get.assert_called()
        called_url = mock_get.call_args[0][0]
        assert called_url.startswith("https://"), f"URL should be upgraded to HTTPS: {called_url}"
        assert result["success"] is True
        print("✓ HTTP to HTTPS upgrade test passed")
    
    # Test 6: Successful fetch and processing
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Documentation</title></head>
            <body>
                <h1>API Documentation</h1>
                <p>This is a test API with the following endpoints:</p>
                <ul>
                    <li>/api/users - Get all users</li>
                    <li>/api/posts - Get all posts</li>
                </ul>
                <script>console.log('should be removed');</script>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = web_fetch("https://docs.example.com", "Extract API endpoints")
        
        assert result["success"] is True
        assert result["url"] == "https://docs.example.com"
        assert "content_summary" in result
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Documentation"
        assert result["metadata"]["from_cache"] is False
        assert "content_length" in result["metadata"]
        print("✓ Successful fetch and processing test passed")
    
    # Test 7: Network timeout error
    with patch('requests.get') as mock_get:
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = web_fetch("https://slow.example.com", "test prompt")
        
        assert result["error"] is True
        assert result["error_code"] == "TIMEOUT"
        assert "timeout" in result["message"].lower()
        print("✓ Network timeout error test passed")
    
    # Test 8: Connection error
    with patch('requests.get') as mock_get:
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = web_fetch("https://unreachable.example.com", "test prompt")
        
        assert result["error"] is True
        assert result["error_code"] == "CONNECTION_ERROR"
        assert "connection error" in result["message"].lower()
        print("✓ Connection error test passed")
    
    # Test 9: HTTP error codes
    with patch('requests.get') as mock_get:
        # Test 403 Forbidden
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        result = web_fetch("https://forbidden.example.com", "test prompt")
        
        assert result["error"] is True
        assert result["error_code"] == "ACCESS_DENIED"
        assert "403" in result["message"]
        print("✓ 403 Forbidden error test passed")
        
        # Test 404 Not Found
        mock_response.status_code = 404
        result = web_fetch("https://notfound.example.com", "test prompt")
        
        assert result["error"] is True
        assert result["error_code"] == "NOT_FOUND"
        assert "404" in result["message"]
        print("✓ 404 Not Found error test passed")
    
    # Test 10: Cross-domain redirect detection
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {'Location': 'https://different-domain.com/article'}
        mock_get.return_value = mock_response
        
        result = web_fetch("https://short.ly/abc123", "test prompt")
        
        assert result["redirect"] is True
        assert result["original_url"] == "https://short.ly/abc123"
        assert result["redirect_url"] == "https://different-domain.com/article"
        assert "different host" in result["message"]
        print("✓ Cross-domain redirect detection test passed")
    
    # Test 11: Same-domain redirect handling
    with patch('requests.get') as mock_get:
        # First call returns redirect
        redirect_response = Mock()
        redirect_response.status_code = 301
        redirect_response.headers = {'Location': '/new-path'}
        
        # Second call returns actual content
        final_response = Mock()
        final_response.status_code = 200
        final_response.text = "<html><head><title>Redirected Page</title></head><body><h1>Final Content</h1></body></html>"
        
        mock_get.side_effect = [redirect_response, final_response]
        
        result = web_fetch("https://example.com/old-path", "test prompt")
        
        assert result["success"] is True
        assert mock_get.call_count == 2
        print("✓ Same-domain redirect handling test passed")
    
    # Test 12: Cache functionality
    clear_cache()
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Cached Page</title></head><body><h1>Test Content</h1></body></html>"
        mock_get.return_value = mock_response
        
        # First request should fetch from web
        result1 = web_fetch("https://cache-test.example.com", "test prompt")
        assert result1["success"] is True
        assert result1["metadata"]["from_cache"] is False
        assert mock_get.call_count == 1
        
        # Second request should use cache
        result2 = web_fetch("https://cache-test.example.com", "different prompt")
        assert result2["success"] is True
        assert result2["metadata"]["from_cache"] is True
        assert mock_get.call_count == 1  # Should not have made another request
        print("✓ Cache functionality test passed")
    
    # Test 13: Empty content handling
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head></head><body></body></html>"
        mock_get.return_value = mock_response
        
        result = web_fetch("https://empty.example.com", "test prompt")
        
        assert result["error"] is True
        assert result["error_code"] == "NO_CONTENT"
        assert "no meaningful content" in result["message"].lower()
        print("✓ Empty content handling test passed")
    
    print("\nAll tests passed! ✅")


def test_cache_functions():
    """Test the cache-related helper functions."""
    
    clear_cache()
    
    # Test cache key generation
    key1 = _generate_cache_key("https://example.com")
    key2 = _generate_cache_key("https://example.com")
    key3 = _generate_cache_key("https://different.com")
    
    assert key1 == key2, "Same URL should generate same cache key"
    assert key1 != key3, "Different URLs should generate different cache keys"
    print("✓ Cache key generation test passed")
    
    # Test cache storage and retrieval
    test_content = ("test html", "test title", time.time())
    _store_in_cache("test_key", test_content)
    
    retrieved = _get_from_cache("test_key")
    assert retrieved == test_content, "Retrieved content should match stored content"
    print("✓ Cache storage and retrieval test passed")
    
    # Test cache expiration (simulated)
    old_time = time.time() - 1000  # 1000 seconds ago (expired)
    old_content = ("old html", "old title", old_time)
    _store_in_cache("expired_key", old_content)
    
    # Manually set old timestamp to simulate expiration
    import tool_webfetch
    tool_webfetch._cache["expired_key"] = (old_content, old_time)
    
    retrieved_expired = _get_from_cache("expired_key")
    assert retrieved_expired is None, "Expired content should not be retrieved"
    print("✓ Cache expiration test passed")
    
    # Test cache clearing
    clear_cache()
    retrieved_after_clear = _get_from_cache("test_key")
    assert retrieved_after_clear is None, "Cache should be empty after clearing"
    print("✓ Cache clearing test passed")
    
    print("\nCache function tests passed! ✅")


if __name__ == "__main__":
    test_web_fetch()
    test_cache_functions()