import requests
from .custom_markdownify import custom_markdownify

def get_url_content(url: str) -> str | None:
    """
    Fetches the content of a given URL and returns it as text.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: The content of the URL as text.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        return None


def get_markdown(url: str) -> str:
    """Fetches the HTML content from a URL and converts it to Markdown.

    Args:
        url (str): The target URL to fetch HTML content from.

    Returns:
        str: A Markdown representation of the fetched HTML.

    Raises:
        ValueError: If the URL is invalid or content cannot be retrieved.
        Exception: For unexpected errors during conversion.
    """
    html = get_url_content(url)
    return custom_markdownify(html)
