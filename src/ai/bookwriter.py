"""
BookWriter — Deep research agent that produces a structured book.

Depends only on:
  - ai.py (same directory)
  - chromadb, ddgs, scrapling (pip packages)
  - /home/ntlpt24/buildmode/agilab/src/mcp_tools/markdown/mrkdwn_analysis.py

Flow:
  Chapter 1 = Overview of query
  Each chapter: search → fetch → store markdown in chromadb
               → MarkdownAgent reads stored docs (headings, sections, content)
               → extract facts → write chapter
  gap_check() → "what's still missing?" → next chapter topic
  Stops when gap_check returns DONE
"""

import asyncio
import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ntlpt24/buildmode/agilab/src/mcp_tools")
sys.path.insert(0, "/home/ntlpt24/buildmode/agilab/src/mcp_tools/markdown")

from ai import LM, agent, AgentResult, TextDelta, ToolCall, ToolResult
import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE  = "http://192.168.170.76:8000"
BOOK_PATH = "book.md"
DB_PATH   = ".chromadb_book"
MAX_CHAPTERS = 12
MAX_STEPS    = 20

# ---------------------------------------------------------------------------
# ChromaDB — single collection "docs", doc_id = URL
# ---------------------------------------------------------------------------
_db: chromadb.ClientAPI = None
_col = None

def _get_col():
    global _db, _col
    if _col is None:
        _db = chromadb.PersistentClient(path=DB_PATH)
        _col = _db.get_or_create_collection("docs")
    return _col

def _store_doc(url: str, markdown: str, title: str = ""):
    col = _get_col()
    col.upsert(
        ids=[url],
        documents=[markdown],
        metadatas=[{"url": url, "title": title}],
    )

def _get_doc(url: str) -> str:
    col = _get_col()
    r = col.get(ids=[url])
    if r["documents"]:
        return r["documents"][0]
    return ""

def _list_docs() -> list[dict]:
    col = _get_col()
    if col.count() == 0:
        return []
    r = col.get()
    return [{"id": r["ids"][i], "title": r["metadatas"][i].get("title","")}
            for i in range(len(r["ids"]))]

# ---------------------------------------------------------------------------
# In-memory book state
# ---------------------------------------------------------------------------
_mind: list[dict] = []   # {"chapter": str, "fact": str, "source": str}
_book: list[dict] = []   # {"num": int, "title": str, "content": str}

# ---------------------------------------------------------------------------
# TOOL 1: search — ddgs + scrapling → chromadb, returns doc IDs
# ---------------------------------------------------------------------------

def _filter_relevant(results: list, topic: str) -> list:
    """Sync LLM call: given ddgs results, return only relevant ones."""
    from openai import OpenAI
    candidates = "\n".join(
        f"[{i+1}] {r.get('title','')} — {r.get('body','')[:150]}\n    URL: {r.get('href','')}"
        for i, r in enumerate(results)
    )
    prompt = f"""Topic: {topic}

Search results:
{candidates}

Reply with ONLY the numbers of relevant results (e.g. "1,3,4"). If none relevant, reply "none"./no_think"""

    client = OpenAI(base_url=f"{API_BASE}/v1", api_key="-")
    model  = client.models.list().data[0].id
    resp   = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,
    )
    raw = re.sub(r'<think>.*?</think>', '', resp.choices[0].message.content or "", flags=re.DOTALL).strip()
    if raw.lower() == "none":
        return []
    indices = []
    for part in re.findall(r'\d+', raw):
        idx = int(part) - 1
        if 0 <= idx < len(results):
            indices.append(idx)
    return [results[i] for i in indices]


def search(query: str, topic: str = "") -> str:
    """Search the web for a query. Filters irrelevant results before fetching.
    Stores fetched markdown in the doc store. Returns stored doc_ids.

    Args:
        query: search query string
        topic: the chapter topic for relevance filtering (use original chapter topic)
    """
    from ddgs import DDGS
    from fetch import scrapling_get

    try:
        results = list(DDGS().text(query, max_results=8))
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return "No results."

    # Filter by relevance before fetching
    filter_topic = topic or query
    relevant = _filter_relevant(results, filter_topic)
    if not relevant:
        return f"No relevant results found for '{query}' (filtered all {len(results)})."
    print(f"  [filter] {len(relevant)}/{len(results)} relevant")

    stored = []
    for r in relevant:
        url   = r.get("href", "")
        title = r.get("title", "")
        if not url:
            continue
        try:
            fetched = scrapling_get(url, extraction_type="markdown")
            if fetched.get("status") == 200 and fetched.get("content"):
                md = "".join(fetched["content"])
                if md.strip():
                    _store_doc(url, md, title)
                    stored.append(f"  [stored] {url}  |  {title}")
        except Exception as e:
            stored.append(f"  [failed] {url} — {e}")

    return f"Stored {len(stored)} docs for '{query}':\n" + "\n".join(stored)

# ---------------------------------------------------------------------------
# TOOL 2: list_docs — show all stored doc IDs
# ---------------------------------------------------------------------------

def list_docs() -> str:
    """List all documents currently stored in the doc store (doc_id = URL).

    Returns:
        Numbered list of doc_ids with titles.
    """
    docs = _list_docs()
    if not docs:
        return "No documents stored yet. Use search() first."
    lines = [f"[{i+1}] {d['id']}\n     title: {d['title']}" for i, d in enumerate(docs)]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# TOOL 3: get_headings — structure overview of a doc
# ---------------------------------------------------------------------------

def get_headings(doc_id: str) -> str:
    """Get all headings from a stored document. Use this first to understand structure.

    Args:
        doc_id: URL of the stored document (from list_docs)
    """
    md = _get_doc(doc_id)
    if not md:
        return f"Doc not found: {doc_id}"
    from mrkdwn_analysis import MarkdownAnalyzer
    try:
        analyzer = MarkdownAnalyzer.from_string(md)
        headers = analyzer.identify_headers().get("Header", [])
        if not headers:
            return "No headings found."
        return "\n".join(f"H{h['level']}: {h['text']}  (line {h.get('line','')})" for h in headers)
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# TOOL 4: get_overview — first ~600 chars = article summary/intro
# ---------------------------------------------------------------------------

def get_overview(doc_id: str) -> str:
    """Get a structural overview of a stored document: stats, all headings with line numbers, and intro paragraphs.
    Always call this before get_section or get_content_by_lines.

    Args:
        doc_id: URL of the stored document
    """
    md = _get_doc(doc_id)
    if not md:
        return f"Doc not found: {doc_id}"
    from mrkdwn_analysis import MarkdownAnalyzer
    try:
        a = MarkdownAnalyzer.from_string(md)
        stats    = a.analyse()
        headers  = a.identify_headers().get("Header", [])
        paragraphs = a.identify_paragraphs().get("Paragraph", [])

        out = ["=== STATS ==="]
        out.append(f"words={stats.get('words',0)}  headers={stats.get('headers',0)}  paragraphs={stats.get('paragraphs',0)}  lists={stats.get('unordered_lists',0)+stats.get('ordered_lists',0)}  tables={stats.get('tables',0)}")

        out.append("\n=== HEADINGS (use exact text in get_section) ===")
        for h in headers:
            out.append(f"  L{h['line']} H{h['level']}: {h['text']}")

        out.append("\n=== INTRO PARAGRAPHS ===")
        for p in paragraphs[:5]:
            out.append(f"  - {p[:200]}")

        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# TOOL 5: get_section — get content under a specific heading
# ---------------------------------------------------------------------------

def get_section(doc_id: str, heading: str) -> str:
    """Get the full content of a section identified by its heading text.

    Args:
        doc_id: URL of the stored document
        heading: Heading text to find (partial match OK)
    """
    md = _get_doc(doc_id)
    if not md:
        return f"Doc not found: {doc_id}"
    lines = md.splitlines()
    start = None
    start_level = None
    for i, line in enumerate(lines):
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m and heading.lower() in m.group(2).lower():
            start = i
            start_level = len(m.group(1))
            break
    if start is None:
        return f"Section '{heading}' not found. Use get_headings() to see available headings."
    # Collect until next heading of same or higher level
    out = [lines[start]]
    for line in lines[start+1:]:
        m = re.match(r'^(#{1,6})\s+', line)
        if m and len(m.group(1)) <= start_level:
            break
        out.append(line)
    return "\n".join(out)[:3000]

# ---------------------------------------------------------------------------
# TOOL 6: get_content_by_lines — precise line range extraction
# ---------------------------------------------------------------------------

def get_content_by_lines(doc_id: str, start: int, end: int) -> str:
    """Get raw content from a document by line range (1-indexed).

    Args:
        doc_id: URL of the stored document
        start: Start line (1-indexed)
        end: End line (inclusive)
    """
    md = _get_doc(doc_id)
    if not md:
        return f"Doc not found: {doc_id}"
    lines = md.splitlines()
    return "\n".join(lines[start-1:end])

# ---------------------------------------------------------------------------
# TOOL 7: store_fact
# ---------------------------------------------------------------------------

def store_fact(chapter: str, fact: str, source: str = "") -> str:
    """Store a key fact or insight extracted from a document, tagged to a chapter.

    Args:
        chapter: chapter title this fact belongs to
        fact: the insight or fact extracted
        source: source doc_id (URL) this came from
    """
    _mind.append({"chapter": chapter, "fact": fact, "source": source})
    return f"Stored fact #{len(_mind)} under '{chapter}'"

# ---------------------------------------------------------------------------
# TOOL 8: recall_facts
# ---------------------------------------------------------------------------

def recall_facts(chapter: str = "") -> str:
    """Recall all stored facts, optionally filtered by chapter.

    Args:
        chapter: chapter title to filter by (empty = return all)
    """
    facts = _mind if not chapter else [f for f in _mind if chapter.lower() in f["chapter"].lower()]
    if not facts:
        return "No facts stored yet."
    lines = [f"[{i+1}] {f['fact']}\n    source: {f['source']}" for i, f in enumerate(facts)]
    return "\n\n".join(lines)

# ---------------------------------------------------------------------------
# TOOL 9: write_chapter
# ---------------------------------------------------------------------------

def write_chapter(title: str, content: str) -> str:
    """Write a completed chapter to the book. Call after storing enough facts.

    Args:
        title: chapter title
        content: full chapter in markdown — cite sources, tag key points [price][brand][spec][opinion]
    """
    num = len(_book) + 1
    _book.append({"num": num, "title": title, "content": content})
    doc = f"## Chapter {num}: {title}\n\n{content}"
    with open(BOOK_PATH, "a") as f:
        f.write(doc + "\n\n---\n\n")
    print(f"\n  [chapter written] #{num}: {title} ({len(content)} chars)")
    return f"Chapter {num} written: '{title}'"

# ---------------------------------------------------------------------------
# All tools
# ---------------------------------------------------------------------------
TOOLS = [search, list_docs, get_headings, get_overview, get_section,
         get_content_by_lines, store_fact, recall_facts, write_chapter]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """/no_think
You are writing ONE chapter of a research book. You have a doc store backed by ChromaDB.

Workflow:
1. search(query) — search web, stores fetched markdown docs automatically. Returns doc_ids.
2. list_docs() — see all available docs
3. get_headings(doc_id) — understand structure of a doc
4. get_overview(doc_id) — read intro/summary of a doc
5. get_section(doc_id, heading) — deep-read a specific section
6. get_content_by_lines(doc_id, start, end) — read precise line ranges
7. store_fact(chapter, fact, source) — save each key insight you extract
8. write_chapter(title, content) — write the chapter using stored facts

Rules:
- Read actual content from docs before storing facts. Don't guess.
- Use get_headings first, then get_section for relevant parts.
- store_fact for every useful insight before writing.
- Tag key points inline: [price] [brand] [spec] [opinion] [data] [warning]
- Cite source URLs inline in the chapter.
- End by calling write_chapter(). Then stop.
"""

GAP_CHECK_PROMPT = """/no_think
You are reviewing a research book in progress.

Given the original question and all chapters written so far:
- If the book fully answers the original question: respond exactly: DONE
- If something important is missing: respond exactly: NEXT: <specific topic for next chapter>

One line only. No explanation.
"""

# ---------------------------------------------------------------------------
# LM singleton
# ---------------------------------------------------------------------------
_lm: LM = None

def _get_lm() -> LM:
    global _lm
    if _lm is None:
        _lm = LM(url=f"{API_BASE}/v1", api_key="-")
    return _lm

# ---------------------------------------------------------------------------
# Run agent cycle
# ---------------------------------------------------------------------------

async def run(system: str, user: str, tools: list = [], max_steps: int = MAX_STEPS) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    final = ""
    async for event in agent(messages, tools=tools, lm=_get_lm(), max_steps=max_steps):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCall):
            args_preview = json.dumps(event.args)[:120]
            print(f"\n  >> {event.name}({args_preview})")
        elif isinstance(event, ToolResult):
            print(f"  << {event.output[:200].replace(chr(10), ' ')}")
        elif isinstance(event, AgentResult):
            final = event.text
    print()
    return final

# ---------------------------------------------------------------------------
# BookWriter
# ---------------------------------------------------------------------------

async def research_chapter(query: str, topic: str):
    print(f"\n  [topic] {topic}")
    await run(
        system=RESEARCH_PROMPT,
        user=f"Original question: {query}\nChapter topic: {topic}",
        tools=TOOLS,
    )


async def gap_check(query: str) -> str:
    if not _book:
        return f"NEXT: Overview of {query}"
    book_text = "\n\n".join(
        f"Chapter {c['num']}: {c['title']}\n{c['content'][:600]}"
        for c in _book
    )
    response = await run(
        system=GAP_CHECK_PROMPT,
        user=f"Original question: {query}\n\nChapters so far:\n{book_text}",
        tools=[],
        max_steps=1,
    )
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


async def main(query: str):
    with open(BOOK_PATH, "w") as f:
        f.write(f"# {query}\n\n")

    print(f"{'='*60}")
    print(f"BOOK WRITER")
    print(f"Query : {query}")
    print(f"Output: {BOOK_PATH}")
    print(f"DB    : {DB_PATH}")
    print(f"{'='*60}")

    chapter_topic = f"Overview: {query}"

    for chapter_num in range(1, MAX_CHAPTERS + 1):
        print(f"\n{'='*60}")
        print(f"CHAPTER {chapter_num}: {chapter_topic}")
        print(f"{'='*60}")

        await research_chapter(query, chapter_topic)

        print(f"\n  [gap check]")
        decision = await gap_check(query)
        print(f"  [gap check] → {decision}")

        if decision.strip() == "DONE" or chapter_num >= MAX_CHAPTERS:
            break

        if "NEXT:" in decision:
            chapter_topic = decision.split("NEXT:", 1)[1].strip()
        else:
            chapter_topic = decision.strip()

    print(f"\n{'='*60}")
    print(f"DONE — {len(_book)} chapters | {len(_mind)} facts")
    print(f"Book saved to: {BOOK_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "good men's watch under 2k"
    asyncio.run(main(query))
