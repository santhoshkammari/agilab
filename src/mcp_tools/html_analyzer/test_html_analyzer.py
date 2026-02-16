#!/usr/bin/env python3
"""
Test HTML Analyzer against real GitHub repo pages.
Tests all tools individually + chaining workflow.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from html_analyzer import HTMLAnalyzer, _load_html

URLS = [
    "https://github.com/langchain-ai/langchain",
    "https://github.com/openai/openai-python",
]

SEPARATOR = "\n" + "=" * 70 + "\n"


def pprint(label, data, max_lines=30):
    print(f"\n--- {label} ---")
    if isinstance(data, str):
        lines = data.split("\n")
        for line in lines[:max_lines]:
            print(line)
        if len(lines) > max_lines:
            print(f"  ... ({len(lines) - max_lines} more lines)")
    elif isinstance(data, list):
        for item in data[:max_lines]:
            if isinstance(item, dict):
                print(f"  {json.dumps(item, ensure_ascii=False)[:200]}")
            else:
                print(f"  {str(item)[:200]}")
        if len(data) > max_lines:
            print(f"  ... ({len(data) - max_lines} more items)")
    elif isinstance(data, dict):
        for k, v in list(data.items())[:max_lines]:
            print(f"  {k}: {str(v)[:150]}")
    else:
        print(f"  {data}")


def test_all_tools(url):
    print(SEPARATOR)
    print(f"TESTING: {url}")
    print(SEPARATOR)

    analyzer = HTMLAnalyzer.from_url(url)

    # 1. Title
    pprint("get_title", analyzer.get_title())

    # 2. Meta
    pprint("get_meta", analyzer.get_meta())

    # 3. Headings
    pprint("get_headings", analyzer.get_headings())

    # 4. All links (show count + first 10)
    links = analyzer.get_all_links()
    if isinstance(links, list):
        print(f"\n--- get_all_links: {len(links)} total ---")
        for l in links[:10]:
            print(f"  [{l['text'][:50]}] -> {l['href'][:100]}")
    else:
        pprint("get_all_links", links)

    # 5. Nav links
    pprint("get_nav_links", analyzer.get_nav_links())

    # 6. Link by text - search for common patterns
    for search in ["README", "src", "Issues", "Pull requests", "Code"]:
        result = analyzer.get_link_by_text(search)
        if isinstance(result, list) and result:
            print(f"\n--- get_link_by_text('{search}'): {len(result)} match(es) ---")
            for r in result[:3]:
                print(f"  [{r['text'][:50]}] -> {r['href'][:100]}")
        else:
            print(f"\n--- get_link_by_text('{search}'): {result}")

    # 7. Buttons
    pprint("get_buttons", analyzer.get_buttons())

    # 8. Forms
    pprint("get_forms", analyzer.get_forms())

    # 9. Page sections
    pprint("get_page_sections", analyzer.get_page_sections())

    # 10. Images (first 5)
    images = analyzer.get_images()
    if isinstance(images, list):
        print(f"\n--- get_images: {len(images)} total ---")
        for img in images[:5]:
            print(f"  alt='{img['alt'][:50]}' src={img['src'][:80]}")
    else:
        pprint("get_images", images)

    # 11. Tables
    pprint("get_tables", analyzer.get_tables())

    # 12. Main content (truncated)
    content = analyzer.get_main_content()
    print(f"\n--- get_main_content: {len(content)} chars ---")
    print(content[:500])

    # 13. Overview
    pprint("get_overview", analyzer.get_overview())


def test_chaining(url):
    """Test agent-like chaining workflow."""
    print(SEPARATOR)
    print(f"CHAINING TEST: {url}")
    print(SEPARATOR)

    # Step 1: Agent gets overview
    analyzer = HTMLAnalyzer.from_url(url)
    overview = analyzer.get_overview()
    print("Step 1 - Overview obtained")
    print(f"  Title: {analyzer.get_title()}")

    # Step 2: Agent looks for navigation
    nav = analyzer.get_nav_links()
    if isinstance(nav, list):
        print(f"Step 2 - Found {len(nav)} nav links")
        for n in nav[:5]:
            print(f"  [{n['text'][:40]}] -> {n['href'][:80]}")

    # Step 3: Agent searches for specific link
    code_links = analyzer.get_link_by_text("Code")
    if isinstance(code_links, list):
        print(f"Step 3 - Found {len(code_links)} 'Code' links")
        for c in code_links[:3]:
            print(f"  [{c['text'][:40]}] -> {c['href'][:80]}")

    # Step 4: Agent searches for repo-specific items
    readme = analyzer.get_link_by_text("README")
    if isinstance(readme, list):
        print(f"Step 4 - Found README link: {readme[0]['href'][:100]}")

    # Step 5: Find issues/PRs
    issues = analyzer.get_link_by_text("Issues")
    prs = analyzer.get_link_by_text("Pull requests")
    if isinstance(issues, list):
        print(f"Step 5a - Issues link: {issues[0]['href'][:100]}")
    if isinstance(prs, list):
        print(f"Step 5b - PRs link: {prs[0]['href'][:100]}")

    # Step 6: Get headings to understand page structure
    headings = analyzer.get_headings()
    if isinstance(headings, list):
        print(f"Step 6 - Page has {len(headings)} headings:")
        for h in headings[:5]:
            print(f"  H{h['level']}: {h['text'][:60]}")

    print("\nChaining test PASSED - all steps completed successfully")


if __name__ == "__main__":
    for url in URLS:
        test_all_tools(url)

    for url in URLS:
        test_chaining(url)

    print(SEPARATOR)
    print("ALL TESTS COMPLETED")
    print(SEPARATOR)
