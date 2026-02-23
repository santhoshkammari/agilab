import pyautogui
import webbrowser
import time
import random
import pyperclip
from typing import List, Dict
from bs4 import BeautifulSoup
import json
from fastmcp import FastMCP

# Lock to serialize browser-based searches (only one pyautogui at a time)
import threading
_search_lock = threading.Lock()

# Global rate limiter: track last search time and enforce cooldown
_last_search_time = 0.0
_search_count = 0
# Cooldown grows with consecutive searches to avoid Google detection
SEARCH_BASE_COOLDOWN = 15.0   # minimum seconds between searches
SEARCH_COOLDOWN_STEP = 5.0    # extra seconds added per consecutive search
SEARCH_COOLDOWN_MAX = 60.0    # cap on cooldown
SEARCH_BURST_RESET = 300.0    # reset counter after 5 min idle

def _jitter(low: float, high: float):
    """Random human-like delay."""
    time.sleep(random.uniform(low, high))


def get_result_from_google_html(html: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Extract search results from Google HTML content

    Args:
        html: Raw HTML content from Google search page
        max_results: Maximum number of results to return

    Returns:
        List of dictionaries with 'url', 'title', and 'description' keys
    """
    soup = BeautifulSoup(html, 'html.parser')
    results = []

    # Find all search result divs (class 'g' or 'tF2Cxc' are common for Google results)
    result_elements = soup.find_all('div', class_=lambda x: x and ('tF2Cxc' in x.split() or 'g' in x.split()))

    for result_element in result_elements[:max_results]:
        # Extract title from h3 tag
        title_element = result_element.find('h3')
        if not title_element:
            continue

        title = title_element.get_text(strip=True)

        # Extract URL from the first anchor tag
        link_element = result_element.find('a')
        if not link_element or not link_element.get('href'):
            continue

        url = link_element['href']

        # Skip non-http(s) URLs
        if not url.startswith('http'):
            continue

        # Extract description from various possible div classes
        description = ""
        # Common description classes: VwiC3b, yXK7lf, lyLwlc, s, st
        desc_classes = ['VwiC3b', 'yXK7lf', 'lyLwlc', 's', 'st']

        for desc_class in desc_classes:
            desc_div = result_element.find('div', class_=lambda x: x and desc_class in str(x))
            if desc_div:
                text = desc_div.get_text(strip=True)
                if text and len(text) > 20:  # Make sure it's substantial text
                    description = text
                    break

        # If still no description, try to find any div with substantial text
        if not description:
            all_divs = result_element.find_all('div')
            for div in all_divs:
                text = div.get_text(strip=True)
                if text and len(text) > 50 and len(text) < 500:
                    description = text
                    break

        results.append({
            "url": url,
            "title": title,
            "description": description
        })

    return results


def _wait_for_rate_limit():
    """Enforce adaptive cooldown between searches to avoid Google detection."""
    global _last_search_time, _search_count

    now = time.time()
    elapsed = now - _last_search_time

    # Reset burst counter if idle long enough
    if elapsed > SEARCH_BURST_RESET:
        _search_count = 0

    # Calculate adaptive cooldown: grows with consecutive searches
    cooldown = min(
        SEARCH_BASE_COOLDOWN + (_search_count * SEARCH_COOLDOWN_STEP),
        SEARCH_COOLDOWN_MAX,
    )

    wait = cooldown - elapsed
    if wait > 0:
        # Add heavy jitter so it doesn't look like a fixed interval
        wait += random.uniform(3.0, 10.0)
        print(f"  [rate-limit] waiting {wait:.1f}s before next search (search #{_search_count + 1})", flush=True)
        time.sleep(wait)

    _search_count += 1
    _last_search_time = time.time()


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform Google search using pyautogui and return parsed results

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results with url, title, and description
    """
    filter_query = query.replace(" ", "+")

    with _search_lock:  # Only one search at a time (pyautogui is not concurrent)
        _wait_for_rate_limit()  # Adaptive cooldown to avoid Google detection

        _jitter(2.0, 5.0)  # Pre-search jitter — human settling in
        webbrowser.open(f"https://www.google.com/search?client=ubuntu-sn&channel=fs&q={filter_query}")
        _jitter(5.0, 10.0)  # Wait for page load — human reading results

        # Get page source
        pyautogui.hotkey('ctrl', 'u')
        _jitter(2.0, 4.0)
        pyautogui.hotkey('ctrl', 'a')
        _jitter(0.5, 1.5)
        pyautogui.hotkey('ctrl', 'c')
        _jitter(0.5, 1.5)
        html_content = pyperclip.paste()

        # Close tabs
        pyautogui.hotkey('ctrl', 'w')
        _jitter(1.0, 2.0)
        pyautogui.hotkey('ctrl', 'w')
        _jitter(2.0, 4.0)  # Post-search cooldown

    # Parse and return results
    return get_result_from_google_html(html_content, max_results)


mcp = FastMCP("Web Search")

@mcp.tool
def search_google(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform Google search using pyautogui and return parsed results

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results with url, title, and description
    """
    return search(query, max_results)


if __name__ == '__main__':
    mcp.run()