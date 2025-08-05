import requests
import time
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import markdownify


# Simple in-memory cache with 15-minute expiration
_cache = {}
CACHE_DURATION = 15 * 60  # 15 minutes in seconds


def web_fetch(url, prompt):
    """
    Fetch content from a URL and process it using AI analysis.
    
    This function fetches web content, converts HTML to markdown, and processes
    it with a specified prompt to extract relevant information.
    
    Args:
        url (str): The URL to fetch content from. Must be a fully-formed valid URL.
                  HTTP URLs are automatically upgraded to HTTPS.
        prompt (str): Instructions for processing the fetched content. Should describe
                     what information to extract or analyze from the content.
    
    Returns:
        dict: Response containing either:
            - Success: {"success": True, "url": str, "content_summary": str, "metadata": dict}
            - Error: {"error": True, "message": str, "url": str, "error_code": str}
            - Redirect: {"redirect": True, "original_url": str, "redirect_url": str, "message": str}
    
    Raises:
        ValueError: If URL format is invalid
        TypeError: If url or prompt are not strings
    """
    # Input validation
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    
    if not url.strip():
        raise ValueError("url cannot be empty")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    
    # Basic URL validation and normalization
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must include protocol (http:// or https://)")
    
    # Upgrade HTTP to HTTPS
    if url.startswith('http://'):
        url = url.replace('http://', 'https://', 1)
    
    # Check cache first
    cache_key = _generate_cache_key(url)
    cached_content = _get_from_cache(cache_key)
    from_cache = False
    
    if cached_content:
        html_content, title, fetch_time = cached_content
        from_cache = True
    else:
        # Fetch content from web
        try:
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; WebFetch Tool)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
                timeout=30,
                allow_redirects=False  # Handle redirects manually
            )
            
            # Handle redirects
            if response.status_code in (301, 302, 303, 307, 308):
                redirect_url = response.headers.get('Location', '')
                if redirect_url:
                    # Check if redirect goes to different host
                    original_host = urlparse(url).netloc
                    if redirect_url.startswith('/'):
                        # Relative redirect - same host
                        redirect_url = urljoin(url, redirect_url)
                    
                    redirect_host = urlparse(redirect_url).netloc
                    if original_host != redirect_host:
                        return {
                            "redirect": True,
                            "original_url": url,
                            "redirect_url": redirect_url,
                            "message": "URL redirected to different host. Please make new request with redirect_url"
                        }
                    else:
                        # Same host redirect - follow it
                        response = requests.get(
                            redirect_url,
                            headers={
                                'User-Agent': 'Mozilla/5.0 (compatible; WebFetch Tool)',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            },
                            timeout=30
                        )
                        url = redirect_url  # Update URL to the final one
            
            if response.status_code == 403:
                return {
                    "error": True,
                    "message": "Access denied - server returned 403 Forbidden",
                    "url": url,
                    "error_code": "ACCESS_DENIED"
                }
            elif response.status_code == 404:
                return {
                    "error": True,
                    "message": "Page not found - server returned 404",
                    "url": url,
                    "error_code": "NOT_FOUND"
                }
            elif response.status_code >= 400:
                return {
                    "error": True,
                    "message": f"HTTP error {response.status_code}",
                    "url": url,
                    "error_code": f"HTTP_{response.status_code}"
                }
            
            html_content = response.text
            fetch_time = time.time()
            
            # Extract title from HTML
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No title"
            except Exception:
                title = "Unable to extract title"
            
            # Cache the content
            _store_in_cache(cache_key, (html_content, title, fetch_time))
            
        except requests.exceptions.Timeout:
            return {
                "error": True,
                "message": "Request timeout - server took too long to respond",
                "url": url,
                "error_code": "TIMEOUT"
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": True,
                "message": "Connection error - unable to reach server",
                "url": url,
                "error_code": "CONNECTION_ERROR"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"Request failed: {str(e)}",
                "url": url,
                "error_code": "REQUEST_ERROR"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Unexpected error: {str(e)}",
                "url": url,
                "error_code": "UNKNOWN_ERROR"
            }
    
    # Convert HTML to markdown
    try:
        markdown_content = _html_to_markdown(html_content)
        if not markdown_content.strip():
            return {
                "error": True,
                "message": "No meaningful content could be extracted from the page",
                "url": url,
                "error_code": "NO_CONTENT"
            }
    except Exception as e:
        return {
            "error": True,
            "message": f"Failed to process HTML content: {str(e)}",
            "url": url,
            "error_code": "CONTENT_PROCESSING_ERROR"
        }
    
    # Process content with AI (simulated - in real implementation would use AI model)
    try:
        ai_response = _process_with_ai(markdown_content, prompt)
    except Exception as e:
        return {
            "error": True,
            "message": f"AI processing failed: {str(e)}",
            "url": url,
            "error_code": "AI_PROCESSING_ERROR"
        }
    
    # Return successful response
    return {
        "success": True,
        "url": url,
        "content_summary": ai_response,
        "metadata": {
            "title": title,
            "fetch_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(fetch_time)),
            "content_length": len(markdown_content),
            "from_cache": from_cache
        }
    }


def _generate_cache_key(url):
    """Generate a cache key for the given URL."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def _get_from_cache(cache_key):
    """Retrieve content from cache if not expired."""
    if cache_key in _cache:
        content, timestamp = _cache[cache_key]
        if time.time() - timestamp < CACHE_DURATION:
            return content
        else:
            # Remove expired entry
            del _cache[cache_key]
    return None


def _store_in_cache(cache_key, content):
    """Store content in cache with timestamp."""
    _cache[cache_key] = (content, time.time())


def _html_to_markdown(html_content):
    """Convert HTML content to clean markdown."""
    try:
        # Parse HTML with BeautifulSoup to clean it first
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Convert to markdown
        markdown = markdownify.markdownify(
            str(soup),
            heading_style="ATX",
            bullets="-",
            strip=['script', 'style']
        )
        
        # Clean up the markdown
        lines = markdown.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that are just whitespace
            if line and not line.isspace():
                cleaned_lines.append(line)
        
        return '\n\n'.join(cleaned_lines)
    
    except Exception as e:
        raise Exception(f"HTML to markdown conversion failed: {str(e)}")


def _process_with_ai(content, prompt):
    """
    Process content with AI model (simulated).
    
    In a real implementation, this would send the content and prompt to an AI model.
    For this implementation, we'll return a simulated response.
    """
    # Simulate AI processing with a basic response
    content_length = len(content)
    word_count = len(content.split())
    
    response = f"""Based on the prompt "{prompt}", here is the analysis of the fetched content:

Content Overview:
- Content length: {content_length} characters
- Estimated word count: {word_count} words

Content Preview (first 500 characters):
{content[:500]}{"..." if len(content) > 500 else ""}

Note: This is a simulated AI response. In a real implementation, this would be processed by an AI model that would analyze the content according to the specific prompt provided and return relevant insights, summaries, or extracted information as requested."""
    
    return response


def clear_cache():
    """Clear the entire cache. Useful for testing."""
    global _cache
    _cache = {}