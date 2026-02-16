"""
HTML Analyzer MCP Tool - Structured traversal of HTML files.

Same pattern as the Markdown MCP but for raw HTML:
  - Input: local HTML file path OR URL (fetched automatically)
  - Exposes tools to traverse the HTML structure: links, nav, forms, buttons, headings, overview
  - An agent can call get_overview -> get_nav_links -> get_link_by_text("src") to navigate
"""

import re
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment

from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class HTMLAnalyzer:
    """Parse an HTML document and expose structured query methods."""

    def __init__(self, html: str, base_url: str = ""):
        self.html = html
        self.base_url = base_url
        self.soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles so they don't pollute text queries
        for tag in self.soup(["script", "style"]):
            tag.decompose()

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_file(cls, file_path: str):
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return cls(text)

    @classmethod
    def from_url(cls, url: str, timeout: int = 30):
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; HTMLAnalyzer/1.0)"
        })
        resp.raise_for_status()
        return cls(resp.text, base_url=url)

    # -- helpers -------------------------------------------------------------

    def _resolve(self, href: str) -> str:
        """Resolve a possibly-relative href against base_url."""
        if not href:
            return ""
        if href.startswith(("http://", "https://", "//")):
            return href
        if self.base_url:
            return urljoin(self.base_url, href)
        return href

    @staticmethod
    def _text(el) -> str:
        return (el.get_text(separator=" ", strip=True) or "").strip()

    @staticmethod
    def _visible_text(el) -> str:
        """Get short visible text (max 200 chars)."""
        t = (el.get_text(separator=" ", strip=True) or "").strip()
        return t[:200] + ("..." if len(t) > 200 else "")

    # -- tool methods --------------------------------------------------------

    def get_title(self) -> str:
        """Return the page <title>."""
        tag = self.soup.find("title")
        return self._text(tag) if tag else "No title"

    def get_meta(self) -> dict:
        """Return common meta tags (description, og:*, etc.)."""
        result = {}
        for m in self.soup.find_all("meta"):
            name = m.get("name") or m.get("property") or ""
            content = m.get("content", "")
            if name and content:
                result[name] = content
        return result or "No meta tags found"

    def get_headings(self) -> list:
        """Extract all headings (h1-h6) with text and hierarchy."""
        headings = []
        for tag in self.soup.find_all(re.compile(r"^h[1-6]$")):
            level = int(tag.name[1])
            text = self._text(tag)
            if text:
                # Check for link inside heading
                link = tag.find("a", href=True)
                entry = {"level": level, "text": text}
                if link:
                    entry["href"] = self._resolve(link["href"])
                headings.append(entry)
        return headings if headings else "No headings found"

    def get_all_links(self) -> list:
        """Extract every <a> link with text and href."""
        links = []
        seen = set()
        for a in self.soup.find_all("a", href=True):
            href = self._resolve(a["href"])
            text = self._text(a)
            key = (href, text)
            if key not in seen and href:
                seen.add(key)
                links.append({"text": text or "(no text)", "href": href})
        return links if links else "No links found"

    def get_nav_links(self) -> list:
        """Extract navigation links from <nav>, header, and common nav patterns."""
        nav_links = []
        seen = set()

        # Strategy 1: <nav> elements
        for nav in self.soup.find_all("nav"):
            for a in nav.find_all("a", href=True):
                href = self._resolve(a["href"])
                text = self._text(a)
                key = (href, text)
                if key not in seen and href:
                    seen.add(key)
                    nav_links.append({
                        "text": text or "(no text)",
                        "href": href,
                        "source": "nav"
                    })

        # Strategy 2: header element links
        for header in self.soup.find_all("header"):
            for a in header.find_all("a", href=True):
                href = self._resolve(a["href"])
                text = self._text(a)
                key = (href, text)
                if key not in seen and href:
                    seen.add(key)
                    nav_links.append({
                        "text": text or "(no text)",
                        "href": href,
                        "source": "header"
                    })

        # Strategy 3: role="navigation"
        for el in self.soup.find_all(attrs={"role": "navigation"}):
            for a in el.find_all("a", href=True):
                href = self._resolve(a["href"])
                text = self._text(a)
                key = (href, text)
                if key not in seen and href:
                    seen.add(key)
                    nav_links.append({
                        "text": text or "(no text)",
                        "href": href,
                        "source": "role=navigation"
                    })

        return nav_links if nav_links else "No navigation links found"

    def get_link_by_text(self, search_text: str) -> list:
        """Find links whose visible text contains search_text (case-insensitive)."""
        search_lower = search_text.lower()
        matches = []
        seen = set()
        for a in self.soup.find_all("a", href=True):
            text = self._text(a)
            href = self._resolve(a["href"])
            if search_lower in text.lower() and href not in seen:
                seen.add(href)
                # Get context: parent's text (truncated)
                parent_text = ""
                if a.parent:
                    parent_text = self._visible_text(a.parent)
                matches.append({
                    "text": text,
                    "href": href,
                    "context": parent_text
                })
        return matches if matches else f"No links matching '{search_text}'"

    def get_buttons(self) -> list:
        """Find all buttons and submit inputs."""
        buttons = []
        # <button> tags
        for btn in self.soup.find_all("button"):
            text = self._text(btn)
            btn_type = btn.get("type", "button")
            entry = {"text": text or "(no text)", "type": btn_type}
            onclick = btn.get("onclick")
            if onclick:
                entry["onclick"] = onclick[:200]
            href = btn.get("data-href") or btn.get("formaction")
            if href:
                entry["href"] = self._resolve(href)
            buttons.append(entry)

        # <input type="submit|button">
        for inp in self.soup.find_all("input", type=re.compile(r"^(submit|button)$", re.I)):
            buttons.append({
                "text": inp.get("value", "(no text)"),
                "type": inp.get("type", "submit"),
                "name": inp.get("name", "")
            })

        # <a> styled as buttons (common pattern: class contains "btn" or "button")
        for a in self.soup.find_all("a", href=True, class_=re.compile(r"btn|button", re.I)):
            text = self._text(a)
            buttons.append({
                "text": text or "(no text)",
                "type": "link-button",
                "href": self._resolve(a["href"])
            })

        return buttons if buttons else "No buttons found"

    def get_forms(self) -> list:
        """Extract all forms with their action, method, and fields."""
        forms = []
        for form in self.soup.find_all("form"):
            action = form.get("action", "")
            if action:
                action = self._resolve(action)
            method = form.get("method", "GET").upper()

            fields = []
            for inp in form.find_all(["input", "textarea", "select"]):
                field = {
                    "tag": inp.name,
                    "name": inp.get("name", ""),
                    "type": inp.get("type", "text") if inp.name == "input" else inp.name,
                }
                placeholder = inp.get("placeholder")
                if placeholder:
                    field["placeholder"] = placeholder
                value = inp.get("value")
                if value:
                    field["value"] = value
                if inp.name == "select":
                    field["options"] = [
                        opt.get_text(strip=True) for opt in inp.find_all("option")
                    ][:10]  # limit
                fields.append(field)

            forms.append({
                "action": action,
                "method": method,
                "fields": fields
            })
        return forms if forms else "No forms found"

    def get_page_sections(self) -> list:
        """Identify major page sections (main, aside, footer, article, etc.)."""
        sections = []
        landmark_tags = ["main", "aside", "footer", "article", "section", "header", "nav"]
        for tag_name in landmark_tags:
            for el in self.soup.find_all(tag_name):
                # Get first heading inside as title
                heading = el.find(re.compile(r"^h[1-6]$"))
                heading_text = self._text(heading) if heading else ""
                # Count links inside
                link_count = len(el.find_all("a", href=True))
                # Short text preview
                preview = self._visible_text(el)[:150]

                aria_label = el.get("aria-label", "")
                role = el.get("role", "")

                sections.append({
                    "tag": tag_name,
                    "aria_label": aria_label,
                    "role": role,
                    "heading": heading_text,
                    "link_count": link_count,
                    "preview": preview
                })
        return sections if sections else "No landmark sections found"

    def get_images(self) -> list:
        """Extract all images with src and alt text."""
        images = []
        for img in self.soup.find_all("img"):
            src = img.get("src", "")
            if src:
                src = self._resolve(src)
            alt = img.get("alt", "")
            images.append({"src": src, "alt": alt})
        return images[:50] if images else "No images found"  # limit to 50

    def get_tables(self) -> list:
        """Extract all tables with headers and row counts."""
        tables = []
        for i, table in enumerate(self.soup.find_all("table"), 1):
            headers = []
            for th in table.find_all("th"):
                headers.append(self._text(th))
            rows = table.find_all("tr")
            row_count = max(0, len(rows) - (1 if headers else 0))

            # First few rows preview
            preview_rows = []
            for tr in rows[1:4] if headers else rows[:3]:  # skip header row
                cells = [self._text(td) for td in tr.find_all(["td", "th"])]
                preview_rows.append(cells)

            tables.append({
                "table_num": i,
                "headers": headers,
                "row_count": row_count,
                "preview_rows": preview_rows
            })
        return tables if tables else "No tables found"

    def get_table_data(self, table_index: int) -> dict:
        """Get full data for a specific table (1-indexed)."""
        all_tables = self.soup.find_all("table")
        if table_index < 1 or table_index > len(all_tables):
            return f"Invalid table index {table_index}. Found {len(all_tables)} tables."

        table = all_tables[table_index - 1]
        headers = [self._text(th) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = [self._text(td) for td in tr.find_all(["td", "th"])]
            if cells and cells != headers:
                rows.append(cells)

        return {
            "headers": headers,
            "rows": rows,
            "total_rows": len(rows)
        }

    def get_main_content(self) -> str:
        """Extract the main content area text (best effort)."""
        # Try <main>, then <article>, then role="main", then largest div
        main = self.soup.find("main") or self.soup.find("article") or \
               self.soup.find(attrs={"role": "main"})

        if not main:
            # Fallback: find the div with the most text
            divs = self.soup.find_all("div")
            if divs:
                main = max(divs, key=lambda d: len(self._text(d)))

        if main:
            text = self._text(main)
            return text[:3000] + ("..." if len(text) > 3000 else "")
        return "Could not identify main content"

    def get_overview(self) -> str:
        """Eagle-eye view of the HTML page: title, structure, stats, key content."""
        title = self.get_title()
        meta = self.get_meta()
        headings = self.get_headings()
        nav_links = self.get_nav_links()
        sections = self.get_page_sections()
        forms = self.get_forms()
        buttons = self.get_buttons()
        all_links = self.get_all_links()
        images = self.get_images()
        tables = self.get_tables()

        # Stats
        link_count = len(all_links) if isinstance(all_links, list) else 0
        nav_count = len(nav_links) if isinstance(nav_links, list) else 0
        heading_count = len(headings) if isinstance(headings, list) else 0
        section_count = len(sections) if isinstance(sections, list) else 0
        form_count = len(forms) if isinstance(forms, list) else 0
        button_count = len(buttons) if isinstance(buttons, list) else 0
        image_count = len(images) if isinstance(images, list) else 0
        table_count = len(tables) if isinstance(tables, list) else 0

        # Description from meta
        desc = ""
        if isinstance(meta, dict):
            desc = meta.get("description", meta.get("og:description", ""))

        # Top headings
        heading_preview = ""
        if isinstance(headings, list):
            for h in headings[:15]:
                indent = "  " * (h["level"] - 1)
                heading_preview += f"{indent}H{h['level']}: {h['text']}\n"

        # Nav preview
        nav_preview = ""
        if isinstance(nav_links, list):
            for nl in nav_links[:20]:
                nav_preview += f"  - [{nl['text']}]({nl['href']})\n"

        # Section preview
        section_preview = ""
        if isinstance(sections, list):
            for s in sections[:10]:
                label = s["aria_label"] or s["heading"] or s["preview"][:60]
                section_preview += f"  <{s['tag']}> {label} ({s['link_count']} links)\n"

        overview = f"""# Page Overview: {title}

## Description
{desc or '(no description)'}

## Stats
- Links: {link_count}
- Navigation links: {nav_count}
- Headings: {heading_count}
- Sections: {section_count}
- Forms: {form_count}
- Buttons: {button_count}
- Images: {image_count}
- Tables: {table_count}

## Page Structure (headings)
{heading_preview or '(none)'}

## Navigation
{nav_preview or '(none)'}

## Sections
{section_preview or '(none)'}
"""
        return overview


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------

def _load_html(source: str) -> HTMLAnalyzer:
    """Load from file path or URL."""
    if source.startswith(("http://", "https://")):
        return HTMLAnalyzer.from_url(source)
    p = Path(source)
    if p.exists():
        return HTMLAnalyzer.from_file(str(p))
    raise FileNotFoundError(f"Not found: {source}")


# ---------------------------------------------------------------------------
# Standalone tool functions (mirror the markdown pattern)
# ---------------------------------------------------------------------------

def html_get_overview(source: str) -> str:
    """Get eagle-eye overview of an HTML page. Source can be a file path or URL."""
    return _load_html(source).get_overview()

def html_get_title(source: str) -> str:
    """Get the page title."""
    return _load_html(source).get_title()

def html_get_meta(source: str):
    """Get meta tags (description, og:*, etc.)."""
    return _load_html(source).get_meta()

def html_get_headings(source: str):
    """Get all headings (h1-h6) with hierarchy."""
    return _load_html(source).get_headings()

def html_get_all_links(source: str):
    """Get every link on the page."""
    return _load_html(source).get_all_links()

def html_get_nav_links(source: str):
    """Get navigation links (from <nav>, <header>, role=navigation)."""
    return _load_html(source).get_nav_links()

def html_get_link_by_text(source: str, search_text: str):
    """Find links whose text contains search_text (case-insensitive)."""
    return _load_html(source).get_link_by_text(search_text)

def html_get_buttons(source: str):
    """Find all buttons and submit inputs."""
    return _load_html(source).get_buttons()

def html_get_forms(source: str):
    """Find all forms with action, method, and fields."""
    return _load_html(source).get_forms()

def html_get_page_sections(source: str):
    """Identify major page sections (main, aside, footer, article, etc.)."""
    return _load_html(source).get_page_sections()

def html_get_images(source: str):
    """Get all images with src and alt."""
    return _load_html(source).get_images()

def html_get_tables(source: str):
    """Get table metadata (headers, row counts, preview)."""
    return _load_html(source).get_tables()

def html_get_table_data(source: str, table_index: int):
    """Get full data for a specific table (1-indexed)."""
    return _load_html(source).get_table_data(table_index)

def html_get_main_content(source: str):
    """Extract the main content area text."""
    return _load_html(source).get_main_content()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("HTML Analysis Server")


@mcp.tool
def mcp_html_get_overview(source: str) -> str:
    """Get eagle-eye overview of an HTML page. Source can be a file path or URL."""
    return html_get_overview(source)


@mcp.tool
def mcp_html_get_title(source: str) -> str:
    """Get the page title."""
    return html_get_title(source)


@mcp.tool
def mcp_html_get_meta(source: str):
    """Get meta tags (description, og:*, etc.)."""
    return html_get_meta(source)


@mcp.tool
def mcp_html_get_headings(source: str):
    """Get all headings (h1-h6) with hierarchy."""
    return html_get_headings(source)


@mcp.tool
def mcp_html_get_all_links(source: str):
    """Get every link on the page."""
    return html_get_all_links(source)


@mcp.tool
def mcp_html_get_nav_links(source: str):
    """Get navigation links (from <nav>, <header>, role=navigation)."""
    return html_get_nav_links(source)


@mcp.tool
def mcp_html_get_link_by_text(source: str, search_text: str):
    """Find links whose text contains search_text (case-insensitive)."""
    return html_get_link_by_text(source, search_text)


@mcp.tool
def mcp_html_get_buttons(source: str):
    """Find all buttons and submit inputs."""
    return html_get_buttons(source)


@mcp.tool
def mcp_html_get_forms(source: str):
    """Find all forms with action, method, and fields."""
    return html_get_forms(source)


@mcp.tool
def mcp_html_get_page_sections(source: str):
    """Identify major page sections (main, aside, footer, article, etc.)."""
    return html_get_page_sections(source)


@mcp.tool
def mcp_html_get_images(source: str):
    """Get all images with src and alt."""
    return html_get_images(source)


@mcp.tool
def mcp_html_get_tables(source: str):
    """Get table metadata (headers, row counts, preview)."""
    return html_get_tables(source)


@mcp.tool
def mcp_html_get_table_data(source: str, table_index: int):
    """Get full data for a specific table (1-indexed)."""
    return html_get_table_data(source, table_index)


@mcp.tool
def mcp_html_get_main_content(source: str):
    """Extract the main content area text."""
    return html_get_main_content(source)


if __name__ == "__main__":
    mcp.run()
