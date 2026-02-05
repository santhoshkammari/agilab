import pyautogui
import webbrowser
import time
import pyperclip
from typing import List, Dict
from bs4 import BeautifulSoup
import json


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


def search_google(query: str, max_results: int = 5) -> List[Dict[str, str]]:
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


if __name__ == '__main__':

    queries = [
        "Python programming",
        "Machine learning basics",
        "Best Python libraries",
        "Data science tutorials",
        "How to use BeautifulSoup",
        "Web scraping with Python",
        "Deep learning frameworks",
        "Natural language processing Python",
        "Python automation tools",
        "Open source Python projects",
        "Stack Overflow most popular questions",
        "How to use ChatGPT API",
        "arXiv latest AI papers",
        "arXiv search machine learning",
        "Python list comprehensions explained",
        "GitHub trending repositories",
        "Best practices for REST APIs",
        "Python vs JavaScript for web development",
        "How to deploy Flask app",
        "Pandas dataframe manipulation",
        "arXiv code search",
        "Stack Overflow Python answers",
        "ChatGPT prompt engineering",
        "arXiv deep learning survey",
        "Python unittest examples",
        "How to use requests library",
        "arXiv NLP review",
        "Stack Overflow error debugging",
        "ChatGPT for code generation",
        "arXiv reinforcement learning",
        "Python asyncio tutorial"
    ]

    all_results = {}

    for query in queries*5:
        print(f"Searching: {query}")
        results = search_google(query, max_results=5)
        all_results[query] = results
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
