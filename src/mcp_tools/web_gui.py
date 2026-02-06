import pyautogui
import webbrowser
import time
import pyperclip
from typing import List, Dict
from bs4 import BeautifulSoup
import json
from fastmcp import FastMCP


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


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform Google search using pyautogui and return parsed results

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results with url, title, and description
    """
    filter_query = query.replace(" ", "+")
    webbrowser.open(f"https://www.google.com/search?client=ubuntu-sn&channel=fs&q={filter_query}")
    time.sleep(2.25)

    # Get page source
    pyautogui.hotkey('ctrl', 'u')
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'c')
    html_content = pyperclip.paste()

    # Close tabs
    pyautogui.hotkey('ctrl', 'w')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'w')

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