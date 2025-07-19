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

# Configure detailed logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.info("Starting browser initialization...")

        if self._initialized and self.page is not None:
            try:
                logger.info(f"Browser already initialized, total time: {time.time() - start_time:.3f}s")
                return
            except:
                # Page/browser is closed, reinitialize
                logger.info("Page/browser is closed, reinitializing...")
                self._initialized = False

        cleanup_start = time.time()
        await self.cleanup()
        logger.info(f"Cleanup took: {time.time() - cleanup_start:.3f}s")

        playwright_start = time.time()
        self.playwright = await async_playwright().start()
        logger.info(f"Playwright start took: {time.time() - playwright_start:.3f}s")

        browser_start = time.time()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled']
        )
        logger.info(f"Browser launch took: {time.time() - browser_start:.3f}s")

        context_start = time.time()
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        logger.info(f"Context creation took: {time.time() - context_start:.3f}s")

        page_start = time.time()
        self.page = await self.context.new_page()
        logger.info(f"Page creation took: {type(self.page)}{time.time() - page_start:.3f}s")

        self._initialized = True
        logger.info(f"TOTAL browser initialization took: {time.time() - start_time:.3f}s")

    async def navigate_to(self, url: str, wait_until: str = "domcontentloaded"):
        """Navigate to a URL with configurable wait_until parameter"""
        nav_start = time.time()
        logger.info(f"Starting navigation to: {url}")

        init_start = time.time()
        await self.initialize()
        logger.info(f"Initialize for navigation took: {time.time() - init_start:.3f}s")

        try:
            # Add retry logic for navigation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    attempt_start = time.time()
                    logger.info(f"Navigation attempt {attempt + 1}/{max_retries}")

                    goto_start = time.time()
                    await self.page.goto(url, wait_until=wait_until, timeout=30000)
                    logger.info(f"Page.goto with wait_until='{wait_until}' took: {time.time() - goto_start:.3f}s")
                    logger.info(f"Navigation attempt {attempt + 1} successful in: {time.time() - attempt_start:.3f}s")
                    break
                except Exception as e:
                    logger.error(
                        f"Navigation attempt {attempt + 1} failed after {time.time() - attempt_start:.3f}s: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    logger.info(f"Retrying navigation in 2 seconds...")
                    await asyncio.sleep(attempt + 1)
        except Exception as e:
            logger.error(f"TOTAL navigation failed after {time.time() - nav_start:.3f}s: {str(e)}")
            raise e

        logger.info(f"TOTAL navigation to {url} took: {time.time() - nav_start:.3f}s")

    async def get_page_html(self):
        """Get the HTML content of the current page"""
        try:
            content_start = time.time()
            logger.info("Getting page HTML content...")
            content = await self.page.content()
            logger.info(
                f"Getting page content took: {time.time() - content_start:.3f}s (content length: {len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            raise e

    async def cleanup(self):
        """Clean up resources without errors"""
        cleanup_start = time.time()
        logger.info("Starting cleanup...")

        try:
            if self.page:
                page_close_start = time.time()
                await self.page.close()
                logger.info(f"Page close took: {time.time() - page_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Page close error: {str(e)}")

        try:
            if self.context:
                context_close_start = time.time()
                await self.context.close()
                logger.info(f"Context close took: {time.time() - context_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Context close error: {str(e)}")

        try:
            if self.browser:
                browser_close_start = time.time()
                await self.browser.close()
                logger.info(f"Browser close took: {time.time() - browser_close_start:.3f}s")
        except Exception as e:
            logger.warning(f"Browser close error: {str(e)}")

        try:
            if self.playwright:
                playwright_stop_start = time.time()
                await self.playwright.stop()
                logger.info(f"Playwright stop took: {time.time() - playwright_stop_start:.3f}s")
        except Exception as e:
            logger.warning(f"Playwright stop error: {str(e)}")

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._initialized = False

        logger.info(f"TOTAL cleanup took: {time.time() - cleanup_start:.3f}s")


class WebSearchTool:
    """Web search implementation using Playwright"""

    def __init__(self, browser: PlaywrightBrowser = None):
        """Initialize the search agent"""
        self.browser = browser

    async def web_search(self, query: str, search_provider: str = 'bing', num_results: int = 10) -> List[
        Dict[str, str]]:
        """Perform a web search and return results"""
        search_start = time.time()
        logger.info(f"Starting web search for query: '{query}' (num_results: {num_results})")

        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f'https://www.{search_provider}.com/search?q={encoded_query}'

            navigation_start = time.time()
            await self.browser.navigate_to(search_url)
            logger.info(f"Navigation to search page took: {time.time() - navigation_start:.3f}s")

            # Get the HTML content
            html_start = time.time()
            html = await self.browser.get_page_html()
            logger.info(f"Getting HTML took: {time.time() - html_start:.3f}s")

            # Extract search results
            extraction_start = time.time()
            if search_provider == 'bing':
                logger.info("Extracting Bing results...")
                search_results = self.get_result_from_bing_html(html, num_results)
            elif search_provider == 'duckduckgo':
                logger.info("Extracting DuckDuckGo results...")
                search_results = self.get_result_from_ddkgo_html(html, num_results)
            else:
                logger.warning(f"Unknown search provider: {search_provider}")
                search_results = []

            logger.info(f"Result extraction took: {time.time() - extraction_start:.3f}s")
            logger.info(f"Found {len(search_results)} results")

            json_start = time.time()
            json_results = json.dumps(search_results)
            logger.info(f"JSON serialization took: {time.time() - json_start:.3f}s")

            logger.info(f"TOTAL web search took: {time.time() - search_start:.3f}s")
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

    @staticmethod
    def get_result_from_ddkgo_html(html_content: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Extract search results from DuckDuckGo HTML content"""
        extract_start = time.time()
        logger.info(
            f"Starting DuckDuckGo result extraction (max_results: {max_results}, HTML length: {len(html_content)})")

        soup_start = time.time()
        soup = BeautifulSoup(html_content, 'html.parser')
        logger.info(f"DuckDuckGo BeautifulSoup parsing took: {time.time() - soup_start:.3f}s")

        results = []

        # Find all result containers
        find_start = time.time()
        result_elements = soup.find_all('article', {'data-testid': 'result'})
        logger.info(
            f"Finding DuckDuckGo result elements took: {time.time() - find_start:.3f}s (found {len(result_elements)} elements)")

        processing_start = time.time()
        for i, result_element in enumerate(result_elements):
            if len(results) >= max_results:
                break

            element_start = time.time()

            # URL
            url_element = result_element.find('a', {'data-testid': 'result-extras-url-link'})
            url = url_element['href'] if url_element else None

            # Title
            title_element = result_element.find('a', {'data-testid': 'result-title-a'})
            title = title_element.get_text(strip=True) if title_element else None

            # Description (Snippet)
            description_element = result_element.find('div', {'data-result': 'snippet'})
            if description_element:
                # Remove date spans if present
                date_span = description_element.find('span', class_=re.compile(r'MILR5XIV'))
                if date_span:
                    date_span.decompose()
                description = description_element.get_text(strip=True)
            else:
                description = None

            if url and title:
                results.append({
                    "url": url,
                    "title": title,
                    "description": description or ""
                })
                logger.debug(
                    f"Processed DuckDuckGo result {i + 1} in {time.time() - element_start:.3f}s: {title[:50]}...")

        logger.info(
            f"Processing {len(result_elements)} DuckDuckGo elements took: {time.time() - processing_start:.3f}s")
        logger.info(
            f"TOTAL DuckDuckGo extraction took: {time.time() - extract_start:.3f}s (extracted {len(results)} results)")
        return results
