# /home/ntlpt59/master/own/deep-researcher/src/tool_web.py
import asyncio
import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import urllib.parse
from duckduckgo_search import DDGS
import pyautogui
import webbrowser
import pyperclip

# Configure detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlaywrightBrowser:
    """A simplified browser interaction manager using Playwright"""

    def __init__(self, headless=True):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self.headless = headless
        self._initialized = False

    async def initialize(self):
        """Initialize the browser if not already done"""
        start_time = time.time()
        logger.debug("Starting browser initialization...")

        if self._initialized and self.page is not None:
            try:
                logger.debug(f"Browser already initialized, total time: {time.time() - start_time:.3f}s")
                return
            except:
                # Page/browser is closed, reinitialize
                logger.debug("Page/browser is closed, reinitializing...")
                self._initialized = False

        cleanup_start = time.time()
        await self.cleanup()
        logger.debug(f"Cleanup took: {time.time() - cleanup_start:.3f}s")

        playwright_start = time.time()
        self.playwright = await async_playwright().start()
        logger.debug(f"Playwright start took: {time.time() - playwright_start:.3f}s")

        browser_start = time.time()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled']
        )
        logger.debug(f"Browser launch took: {time.time() - browser_start:.3f}s")

        context_start = time.time()
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        logger.debug(f"Context creation took: {time.time() - context_start:.3f}s")

        page_start = time.time()
        self.page = await self.context.new_page()
        logger.debug(f"Page creation took: {type(self.page)}{time.time() - page_start:.3f}s")

        self._initialized = True
        logger.debug(f"TOTAL browser initialization took: {time.time() - start_time:.3f}s")

    async def navigate_to(self, url: str, wait_until: str = "domcontentloaded"):
        """Navigate to a URL with configurable wait_until parameter"""
        nav_start = time.time()
        logger.debug(f"Starting navigation to: {url}")

        init_start = time.time()
        await self.initialize()
        logger.debug(f"Initialize for navigation took: {time.time() - init_start:.3f}s")

        try:
            # Add retry logic for navigation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    attempt_start = time.time()
                    logger.debug(f"Navigation attempt {attempt + 1}/{max_retries}")

                    goto_start = time.time()
                    await self.page.goto(url, wait_until=wait_until, timeout=30000)
                    logger.debug(f"Page.goto with wait_until='{wait_until}' took: {time.time() - goto_start:.3f}s")
                    logger.debug(f"Navigation attempt {attempt + 1} successful in: {time.time() - attempt_start:.3f}s")
                    break
                except Exception as e:
                    logger.error(
                        f"Navigation attempt {attempt + 1} failed after {time.time() - attempt_start:.3f}s: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    logger.debug(f"Retrying navigation in 2 seconds...")
                    await asyncio.sleep(attempt + 1)
        except Exception as e:
            logger.error(f"TOTAL navigation failed after {time.time() - nav_start:.3f}s: {str(e)}")
            raise e

        logger.debug(f"TOTAL navigation to {url} took: {time.time() - nav_start:.3f}s")

    async def get_page_html(self):
        """Get the HTML content of the current page"""
        try:
            content_start = time.time()
            logger.debug("Getting page HTML content...")
            content = await self.page.content()
            logger.debug(
                f"Getting page content took: {time.time() - content_start:.3f}s (content length: {len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            raise e

    async def cleanup(self):
        """Clean up resources without errors"""
        cleanup_start = time.time()
        logger.debug("Starting cleanup...")

        try:
            if self.page:
                page_close_start = time.time()
                await self.page.close()
                logger.debug(f"Page close took: {time.time() - page_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Page close error: {str(e)}")

        try:
            if self.context:
                context_close_start = time.time()
                await self.context.close()
                logger.debug(f"Context close took: {time.time() - context_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Context close error: {str(e)}")

        try:
            if self.browser:
                browser_close_start = time.time()
                await self.browser.close()
                logger.debug(f"Browser close took: {time.time() - browser_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Browser close error: {str(e)}")

        try:
            if self.playwright:
                playwright_stop_start = time.time()
                await self.playwright.stop()
                logger.debug(f"Playwright stop took: {time.time() - playwright_stop_start:.3f}s")
        except Exception as e:
            logger.warning(f"Playwright stop error: {str(e)}")

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._initialized = False

        logger.debug(f"TOTAL cleanup took: {time.time() - cleanup_start:.3f}s")


class WebSearchTool:
    """Web search implementation using Playwright"""

    def __init__(self, browser: PlaywrightBrowser = None):
        """Initialize the search agent"""
        self.browser = browser

    async def web_search(self, query: str, search_provider: str = 'bing', max_results: int = 10) -> str:
        """Perform a web search and return results"""
        search_start = time.time()
        logger.debug(f"Starting web search for query: '{query}' (num_results: {max_results})")

        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f'https://www.{search_provider}.com/search?q={encoded_query}'

            navigation_start = time.time()
            await self.browser.navigate_to(search_url)
            logger.debug(f"Navigation to search page took: {time.time() - navigation_start:.3f}s")

            # Get the HTML content
            html_start = time.time()
            html = await self.browser.get_page_html()
            logger.debug(f"Getting HTML took: {time.time() - html_start:.3f}s")

            # Extract search results
            extraction_start = time.time()
            if search_provider == 'bing':
                logger.debug("Extracting Bing results...")
                search_results = self.get_result_from_bing_html(html, max_results)
            else:
                logger.warning(f"Unknown search provider: {search_provider}")
                search_results = []

            logger.debug(f"Result extraction took: {time.time() - extraction_start:.3f}s")
            logger.debug(f"Found {len(search_results)} results")

            json_start = time.time()
            json_results = json.dumps(search_results)
            logger.debug(f"JSON serialization took: {time.time() - json_start:.3f}s")

            logger.debug(f"TOTAL web search took: {time.time() - search_start:.3f}s")
            return json_results

        except Exception as e:
            logger.error(f"Search error after {time.time() - search_start:.3f}s: {str(e)}")
            return json.dumps([])

    @staticmethod
    def get_result_from_bing_html(html: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Extract search results from Bing HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        result_elements = soup.find_all('li', class_='b_algo')

        for result_element in result_elements[:max_results]:
            title_header = result_element.find('h2')
            if title_header:
                title_link = title_header.find('a')
                if title_link and title_link.get('href'):
                    url = title_link['href']
                    title = title_link.get_text(strip=True)

                    description = ""
                    caption_div = result_element.find('div', class_='b_caption')
                    if caption_div:
                        p_tag = caption_div.find('p')
                        if p_tag:
                            description = p_tag.get_text(strip=True)

                    results.append({
                        "url": url,
                        "title": title,
                        "description": description
                    })
        return results

class DDGSearchTool:
    def __init__(self,browser=None):
        self.ddgs = DDGS()
        self.browser = browser
    
    def web_search(self, query: str, max_results: int = 10) -> str:
        """Search using DuckDuckGo and return results as JSON string"""
        try:
            logger.debug(f"Starting DDG search for query: '{query}' (max_results: {max_results})")
            
            search_results = []
            results = self.ddgs.text(query, max_results=max_results)
            
            for result in results:
                search_results.append({
                    "url": result.get("href", ""),
                    "title": result.get("title", ""),
                    "description": result.get("body", "")
                })
            
            logger.debug(f"Found {len(search_results)} DDG results")
            return json.dumps(search_results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"DDG search error: {str(e)}")
            return json.dumps([])

class GuiSearchTool:
    def __init__(self,browser=None):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        self.browser = None
    
    async def web_search(self, query: str, max_results: int = 10) -> str:
        """Search using GUI automation with Chrome and return HTML content"""
        try:
            logger.debug(f"Starting GUI search for query: '{query}'")
            
            # Open browser with search URL directly
            webbrowser.open("", new=1)
            time.sleep(0.5)  # Longer wait for Firefox

            pyautogui.hotkey('ctrl','l')
            time.sleep(0.1)

            pyautogui.typewrite(query)
            time.sleep(0.1)  # Longer wait for Firefox
            pyautogui.press('enter')
            time.sleep(1)  # Longer wait for Firefox

            
            # Get page source (Ctrl+U for Firefox)
            pyautogui.hotkey('ctrl', 'u')
            time.sleep(0.1)
            
            # Select all content (Ctrl+A)
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.1)
            
            # Copy content (Ctrl+C)
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.1)
            
            # Get clipboard content
            html_content = pyperclip.paste()
            
            # Close the view-source tab (Ctrl+W)
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(0.1)
            
            # Close the search results tab (Ctrl+W)
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(0.1)
            
            # Process HTML to extract search results
            search_results = self._extract_google_results(html_content, max_results)
            
            logger.debug(f"Found {len(search_results)} GUI search results")
            return json.dumps(search_results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"GUI search error: {str(e)}")
            return json.dumps([])
    
    def _extract_google_results(self, html_content: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Extract search results from Google HTML content"""
        with open('html.txt','w') as f:
            f.write(html_content)
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # Try multiple selectors as Google changes its structure frequently
            selectors_to_try = [
                'div.g',                    # Classic Google results
                'div.tF2Cxc',              # New Google results container
                'div[jscontroller="SC7lYd"]',  # Results with JS controller
                'div.MjjYud',              # Another common container
                'div[data-hveid]',         # Results with hveid attribute
                'div.Wt5Tfe'               # Alternative container
            ]
            
            result_elements = []
            for selector in selectors_to_try:
                result_elements = soup.select(selector)
                if result_elements:
                    logger.debug(f"Found {len(result_elements)} results using selector: {selector}")
                    break
            
            # If no structured results found, try to find any links that look like search results
            if not result_elements:
                logger.debug("No structured results found, trying alternative extraction")
                # Look for links that might be search results
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '')
                    # Skip internal Google links and ads
                    if (href.startswith('http') and 
                        'google.com' not in href and 
                        'googleadservices.com' not in href and
                        'youtube.com' not in href and
                        link.get_text(strip=True)):
                        
                        title = link.get_text(strip=True)
                        if len(title) > 10:  # Reasonable title length
                            results.append({
                                "url": href,
                                "title": title,
                                "description": ""
                            })
                            if len(results) >= max_results:
                                break
            else:
                # Process structured results
                for result_element in result_elements[:max_results]:
                    title = ""
                    url = ""
                    description = ""
                    
                    # Try different ways to find title and URL
                    # Method 1: Look for h3 with parent link
                    title_element = result_element.find('h3')
                    if title_element:
                        title = title_element.get_text(strip=True)
                        link_element = title_element.find_parent('a')
                        if link_element:
                            url = link_element.get('href', '')
                    
                    # Method 2: Look for any link in the result
                    if not url:
                        link_element = result_element.find('a', href=True)
                        if link_element:
                            url = link_element.get('href', '')
                            if not title:
                                title = link_element.get_text(strip=True)
                    
                    # Method 3: Look for cite elements or URL displays
                    if not url:
                        cite_element = result_element.find('cite')
                        if cite_element:
                            url = cite_element.get_text(strip=True)
                    
                    # Extract description from various possible containers
                    desc_selectors = [
                        'span[data-ved]',
                        'div.VwiC3b',
                        'div.s3v9rd',
                        'div.IsZvec',
                        'span.aCOpRe',
                        'span.st'
                    ]
                    
                    for desc_selector in desc_selectors:
                        desc_element = result_element.select_one(desc_selector)
                        if desc_element:
                            desc_text = desc_element.get_text(strip=True)
                            if len(desc_text) > 20:
                                description = desc_text
                                break
                    
                    # If no description found, get any text that looks like a description
                    if not description:
                        text_elements = result_element.find_all(text=True)
                        for text in text_elements:
                            text = text.strip()
                            if len(text) > 30 and len(text) < 200:
                                description = text
                                break
                    
                    # Clean up URL (remove Google redirect prefixes)
                    if url.startswith('/url?'):
                        # Extract actual URL from Google redirect
                        from urllib.parse import urlparse, parse_qs
                        try:
                            parsed = urlparse(url)
                            if 'url' in parse_qs(parsed.query):
                                url = parse_qs(parsed.query)['url'][0]
                        except:
                            pass
                    
                    if title and url:
                        results.append({
                            "url": url,
                            "title": title,
                            "description": description
                        })
            
            logger.debug(f"Successfully extracted {len(results)} search results")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error extracting Google results: {str(e)}")
            return []

# Global browser instance for reuse
_browser_instance = None

async def _get_browser():
    """Get or create browser instance"""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = PlaywrightBrowser()
        await _browser_instance.initialize()
    return _browser_instance

def search_web(query: str, max_results: int = 5) -> str:
    """Async wrapper for playwright web search - synchronous interface"""
    async def _search():
        browser = await _get_browser()
        search_tool = WebSearchTool(browser)
        return await search_tool.web_search(query, 'bing', max_results)
    
    return asyncio.run(_search())

def search_web_ddg(query: str, max_results: int = 5) -> str:
    """Synchronous DDG search wrapper"""
    search_tool = DDGSearchTool()
    return search_tool.web_search(query, max_results)

def search_web_gui(query: str, max_results: int = 5) -> str:
    """Synchronous GUI search wrapper"""
    search_tool = GuiSearchTool()
    async def _run():
        return await search_tool.web_search(query,max_results)
    return asyncio.run(_run())

tool_functions = {
    "search_web": search_web,
    # "search_web_ddg": search_web_ddg,
    # "search_web_gui": search_web_gui,
}

if __name__ == '__main__':
    # Test the wrapper functions
    result = search_web('who is modi?')
    print(result)
