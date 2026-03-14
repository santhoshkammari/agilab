import asyncio
import base64
import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import urllib.parse
from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PlaywrightBrowser:
    def __init__(self, headless=True):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self.headless = headless
        self._initialized = False

    async def initialize(self):
        start_time = time.time()
        logger.debug("Starting browser initialization...")

        if self._initialized and self.page is not None:
            try:
                logger.debug(
                    f"Browser already initialized, total time: {time.time() - start_time:.3f}s"
                )
                return
            except:
                logger.debug("Page/browser is closed, reinitializing...")
                self._initialized = False

        await self.cleanup()

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.firefox.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
        )
        self.page = await self.context.new_page()
        self._initialized = True
        logger.debug(
            f"TOTAL browser initialization took: {time.time() - start_time:.3f}s"
        )

    async def navigate_to(self, url: str, wait_until: str = "domcontentloaded"):
        nav_start = time.time()
        logger.debug(f"Starting navigation to: {url}")

        await self.initialize()

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    attempt_start = time.time()
                    logger.debug(f"Navigation attempt {attempt + 1}/{max_retries}")

                    await self.page.goto(url, wait_until=wait_until, timeout=5000)
                    logger.debug(
                        f"Navigation attempt {attempt + 1} successful in: {time.time() - attempt_start:.3f}s"
                    )
                    break
                except Exception as e:
                    logger.error(f"Navigation attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"TOTAL navigation failed: {str(e)}")
            raise e

        logger.debug(f"TOTAL navigation to {url} took: {time.time() - nav_start:.3f}s")

    async def get_page_html(self):
        try:
            content = await self.page.content()
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            raise e

    async def cleanup(self):
        try:
            if self.page:
                await self.page.close()
        except Exception as e:
            logger.warning(f"Page close error: {str(e)}")

        try:
            if self.context:
                await self.context.close()
        except Exception as e:
            logger.warning(f"Context close error: {str(e)}")

        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            logger.warning(f"Browser close error: {str(e)}")

        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.warning(f"Playwright stop error: {str(e)}")

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._initialized = False


class DuckDuckGoSearchTool:
    def __init__(self, browser: PlaywrightBrowser = None):
        self.browser = browser

    async def web_search(self, query: str, max_results: int = 10) -> str:
        search_start = time.time()
        logger.debug(
            f"Starting DuckDuckGo search for query: '{query}' (num_results: {max_results})"
        )

        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://duckduckgo.com/?q={encoded_query}&ia=web"

            await self.browser.navigate_to(search_url)
            time.sleep(3)

            html = await self.browser.get_page_html()
            search_results = self.get_result_from_duckduckgo_html(html, max_results)

            logger.debug(f"Found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    @staticmethod
    def get_result_from_duckduckgo_html(
        html: str, max_results: int = 10
    ) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        result_elements = soup.find_all("div", class_="result__body")

        for result_element in result_elements[:max_results]:
            title_link = result_element.find("a", class_="result__a")
            if title_link and title_link.get("href"):
                url = title_link["href"]
                title = title_link.get_text(strip=True)

                description = ""
                desc_element = result_element.find("a", class_="result__snippet")
                if desc_element:
                    description = desc_element.get_text(strip=True)

                results.append({"url": url, "title": title, "description": description})

        if not results:
            legacy_results = soup.find_all("div", class_="result")
            for result_element in legacy_results[:max_results]:
                title_header = result_element.find("a", class_="result__a")
                if title_header and title_header.get("href"):
                    url = title_header["href"]
                    title = title_header.get_text(strip=True)

                    description = ""
                    desc_div = result_element.find("div", class_="result__snippet")
                    if desc_div:
                        description = desc_div.get_text(strip=True)

                    results.append(
                        {"url": url, "title": title, "description": description}
                    )

        return results


_browser_instance = None


async def _get_browser():
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = PlaywrightBrowser()
        await _browser_instance.initialize()
    return _browser_instance


def search_web_duckduckgo(query: str, max_results: int = 5) -> str:
    async def _search():
        browser = await _get_browser()
        search_tool = DuckDuckGoSearchTool(browser)
        return await search_tool.web_search(query, max_results)

    return asyncio.run(_search())


async def async_web_search_duckduckgo(query: str, max_results: int = 5):
    browser = await _get_browser()
    search_tool = DuckDuckGoSearchTool(browser)
    return await search_tool.web_search(query, max_results)


mcp = FastMCP("DuckDuckGo Search Server")


@mcp.tool
async def web_search_duckduckgo(query: str, max_results: int = 5) -> dict:
    try:
        results = await async_web_search_duckduckgo(query, max_results)
        result = {"query": query, "results": results}
    except Exception as e:
        result = {"error": str(e)}
    return result


tool_functions = {
    "async_web_search_duckduckgo": async_web_search_duckduckgo,
}

if __name__ == "__main__":
    mcp.run()
