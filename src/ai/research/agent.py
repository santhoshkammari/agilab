"""
Research Agent — autonomous tool-driven research using ChromaDB as working memory.
All intelligence lives in the system prompt. Tools are generic primitives.

Uses OpenAI-compatible tool calling API (vLLM with --enable-auto-tool-choice).
"""
import sys
import os
import json
import inspect

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_tools'))

from ai import LM, configure
from research.memory import (
    init as init_memory,
    memory_store, memory_search, memory_exists,
    memory_delete, memory_update
)

# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 10) -> str:
    """Search the web and return URLs with titles and descriptions.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 10)

    Returns:
        Search results as text, each with url, title and description.
    """
    from ddgs import DDGS

    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        url = r.get('href', '')
        title = r.get('title', '')
        desc = r.get('body', '')
        lines.append(f"[{i}] {title}\n    URL: {url}\n    {desc}")
    return "\n\n".join(lines)


def web_fetch(url: str) -> str:
    """Fetch a URL, convert to markdown, and automatically store in the 'research_data' ChromaDB collection.

    The full markdown content is stored in the collection — it does NOT appear in your context.
    Use memory_search to query the stored content later.

    Args:
        url: The URL to fetch and store

    Returns:
        Confirmation message with URL and content size. The content itself is in ChromaDB.
    """
    from fetch import scrapling_get

    if memory_exists("research_data", url):
        return f"Already fetched: {url} (exists in research_data)"

    result = scrapling_get(url, extraction_type="markdown")

    if not result.get("content") or result.get("status") != 200:
        return f"Failed to fetch {url}: status={result.get('status', 0)}"

    markdown = "".join(result["content"])
    if not markdown.strip():
        return f"Fetched {url} but content was empty."

    memory_store(
        collection="research_data",
        text=markdown,
        metadata={"url": url, "char_count": len(markdown)},
        id=url
    )

    return f"Added to research_data: {url} ({len(markdown)} chars)"


# ---------------------------------------------------------------------------
# File tools — for progress.md
# ---------------------------------------------------------------------------

_progress_path = "progress.md"


def write_progress(content: str) -> str:
    """Write/overwrite the progress file (tqdm-style progress tracking).

    Args:
        content: The full progress content. Example:
            [████████░░░░░░░░░░░░] 8/15
            Strategy: following linked leads
            Last: "RAG blog" ✓ unique
            Gaps: AI safety, multimodal
            Queue: 3 leads pending

    Returns:
        Confirmation message.
    """
    with open(_progress_path, 'w') as f:
        f.write(content)
    return f"Progress updated."


def read_progress() -> str:
    """Read the current progress file to see where you left off.

    Returns:
        Contents of progress.md, or 'No progress file found' if first run.
    """
    try:
        with open(_progress_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "No progress file found. This is the first run."


# ---------------------------------------------------------------------------
# OpenAI-compatible Tool Calling Engine
# ---------------------------------------------------------------------------

def _build_tool_schemas(tools: list) -> list[dict]:
    """Build OpenAI-style tool schemas from function signatures + docstrings."""
    from transformers.utils import get_json_schema
    schemas = []
    for fn in tools:
        schema = get_json_schema(fn)
        # Remove 'return' field — vLLM doesn't expect it
        schema.get("function", {}).pop("return", None)
        schemas.append(schema)
    return schemas


def _execute_tool(fn, arguments: dict) -> str:
    """Execute a tool function with the given arguments."""
    try:
        result = fn(**arguments)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_agent_cycle(
    lm: LM,
    system_prompt: str,
    user_input: str,
    tools: list,
    max_turns: int = 30,
    verbose: bool = True,
) -> str:
    """Run one agent cycle using OpenAI-compatible tool calling API.

    Returns:
        The agent's final text response.
    """
    from openai import OpenAI

    client = OpenAI(api_key=lm.api_key or "EMPTY", base_url=f"{lm.api_base}/v1")

    # Get model name from server if not set
    model = lm.model or client.models.list().data[0].id

    tool_map = {fn.__name__: fn for fn in tools}
    tool_schemas = _build_tool_schemas(tools)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    final_response = ""

    for turn in range(max_turns):
        # Call LLM with tools
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            temperature=lm.defaults.get("temperature", 0.7),
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        if not tool_calls:
            # No tool calls — agent is done
            import re
            content = (msg.content or "").strip()
            # Strip <think>...</think> from Qwen3 responses
            final_response = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            if verbose:
                print(f"  [turn {turn+1}] Final response (no tools)", file=sys.stderr)
            break

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}

            if verbose:
                args_preview = json.dumps(args)[:100]
                print(f"  [turn {turn+1}] tool: {name}({args_preview})", file=sys.stderr)

            fn = tool_map.get(name)
            if fn:
                result = _execute_tool(fn, args)
            else:
                result = f"Error: Unknown tool '{name}'. Available: {list(tool_map.keys())}"

            if verbose:
                print(f"  [result] {result[:150]}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    else:
        final_response = "Max tool-call turns reached for this cycle."

    return final_response


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent. You work in SHORT BURSTS — do ONE meaningful cycle per invocation, then stop.

## Your Memory (ChromaDB collections)
You have 5 collections that persist between invocations:
- research_data: Raw fetched content (url → full markdown). web_fetch auto-stores here.
- mind: Your observations — absorbed ideas, concepts, author stances, uniqueness notes.
- searches: Log of every search query you've run and what it yielded.
- queue: Leads to explore — URLs from blogs, query ideas from absorbed concepts.
- state: Your progress metadata.

## Each Invocation — Follow This Cycle

### 1. READ STATE
- Call read_progress() to see where you are
- If target is reached → respond with exactly "DONE" and stop
- If this is the first run, initialize progress

### 2. DECIDE NEXT ACTION
- Check queue: memory_search("queue", "pending leads", n=5)
  → If queue has leads, pick the best one
  → If queue is empty, go to gap detection
- Detect concept gaps: probe "mind" with different concept areas
  → memory_search("mind", "concept X", n=5) — few/no results = GAP
  → Generate query targeting the biggest gap
- Before running query: memory_search("searches", your_query, n=3)
  → If very similar query exists with poor results, modify it

### 3. EXECUTE
- web_search(query) → get URLs and snippets
- For each promising URL:
  - web_fetch(url) → auto-stored (returns confirmation, not content)
- Log: memory_store("searches", query_text, {"results_count": N, "new_found": N, "already_seen": N})

### 4. ABSORB (for each newly fetched URL)
- Query research_data to understand what you fetched:
  - memory_search("research_data", "main idea of [topic]", n=2)
  - memory_search("research_data", "links references", n=2)
  - memory_search("research_data", "author perspective", n=2)
- Store observations in mind:
  memory_store("mind", "Blog by X. KEY IDEA: ... CONCEPTS: ... LINKS: ...",
               {"source_url": "url", "concepts": "a,b,c", "type": "absorption"})
- Extract leads into queue:
  memory_store("queue", "lead description", {"type": "query_candidate", "emerged_from": "url", "reason": "why"})

### 5. ASSESS UNIQUENESS
- memory_search("mind", "key concepts from this content", n=10)
- High similarity to many existing → NOT unique, don't count
- New cluster or perspective → UNIQUE, count it

### 6. UPDATE PROGRESS
- write_progress() with tqdm-style format:
  [████░░░░░░░░░░░░░░░░] 4/15
  Strategy: <current approach>
  Last: "<title>" ✓ unique / ✗ duplicate
  Gaps: <uncovered areas>
  Queue: <N> leads pending
- Then STOP. Respond with a brief status like "Completed cycle. 4/15."

## Rules
- ONE cycle per invocation. Don't try to do everything at once.
- Always check memory_exists before fetching.
- web_fetch auto-stores content. You never see full text in your context.
- When queries yield diminishing returns, CHANGE STRATEGY.
- If you can't find more after many attempts, respond "EXHAUSTED".
- When target reached, respond "DONE"."""


# ---------------------------------------------------------------------------
# ResearchAgent class
# ---------------------------------------------------------------------------

class ResearchAgent:
    """Autonomous research agent with ChromaDB memory.

    Usage:
        agent = ResearchAgent(api_base="http://192.168.170.76:8000")
        agent.run("Find 15 unique blogs on GenAI with diverse perspectives")
    """

    def __init__(
        self,
        model: str = "",
        api_base: str = "http://192.168.170.76:8000",
        api_key: str = "-",
        chromadb_path: str = ".chromadb",
        progress_path: str = "progress.md",
        max_cycles: int = 200,
        max_turns_per_cycle: int = 30,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        self.max_cycles = max_cycles
        self.max_turns_per_cycle = max_turns_per_cycle
        self.verbose = verbose

        global _progress_path
        _progress_path = progress_path

        # Init ChromaDB
        init_memory(chromadb_path)

        # Init LM
        self.lm = LM(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
        )

        # All tools
        self.tools = [
            memory_store, memory_search, memory_exists,
            memory_delete, memory_update,
            web_search, web_fetch,
            write_progress, read_progress,
        ]

    def run(self, task: str):
        """Run the research agent on a task."""
        print(f"{'='*60}")
        print(f"RESEARCH AGENT")
        print(f"Task: {task}")
        print(f"Max cycles: {self.max_cycles}")
        print(f"{'='*60}\n")

        for cycle in range(1, self.max_cycles + 1):
            print(f"\n--- Cycle {cycle}/{self.max_cycles} ---")

            try:
                response = run_agent_cycle(
                    lm=self.lm,
                    system_prompt=RESEARCH_SYSTEM_PROMPT,
                    user_input=task,
                    tools=self.tools,
                    max_turns=self.max_turns_per_cycle,
                    verbose=self.verbose,
                )
            except Exception as e:
                print(f"  Error in cycle {cycle}: {e}")
                continue

            print(f"  Agent: {response[:300]}")

            # Show progress file
            try:
                with open(_progress_path, 'r') as f:
                    progress = f.read().strip()
                    if progress:
                        print(f"\n  {progress.replace(chr(10), chr(10) + '  ')}")
            except FileNotFoundError:
                pass

            if "DONE" in response:
                print(f"\n{'='*60}")
                print(f"COMPLETED after {cycle} cycles!")
                print(f"{'='*60}")
                return response

            if "EXHAUSTED" in response:
                print(f"\n{'='*60}")
                print(f"EXHAUSTED after {cycle} cycles")
                print(f"{'='*60}")
                return response

        print(f"\n{'='*60}")
        print(f"Reached max cycles ({self.max_cycles})")
        print(f"{'='*60}")
        return "MAX_CYCLES_REACHED"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument("task", nargs="?",
                        default="Find 5 unique blogs on GenAI with diverse perspectives",
                        help="Research task")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--api-base", default="http://192.168.170.76:8000")
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--chromadb-path", default=".chromadb")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    agent = ResearchAgent(
        model=args.model,
        api_base=args.api_base,
        max_cycles=args.max_cycles,
        chromadb_path=args.chromadb_path,
        verbose=args.verbose,
    )
    agent.run(args.task)
