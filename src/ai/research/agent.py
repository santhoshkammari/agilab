"""
Research Agent — autonomous tool-driven research using ChromaDB as working memory.
All intelligence lives in the system prompt. Tools are generic primitives.

Uses text-based tool calling (<tool_call> tags) — works with any LLM, no special
vLLM tool-call API flags needed.
"""
import sys
import os
import json
import re
import asyncio
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
# Text-based Tool Calling Engine
# ---------------------------------------------------------------------------

def _build_tool_descriptions(tools: list) -> str:
    """Build tool descriptions for the system prompt from function signatures."""
    lines = []
    for fn in tools:
        sig = inspect.signature(fn)
        params = []
        for name, p in sig.parameters.items():
            ptype = p.annotation.__name__ if p.annotation != inspect.Parameter.empty else "any"
            default = f" = {p.default!r}" if p.default != inspect.Parameter.empty else ""
            params.append(f"{name}: {ptype}{default}")
        params_str = ", ".join(params)

        doc = fn.__doc__ or ""
        # Take first paragraph of docstring
        first_para = doc.split("\n\n")[0].strip().replace("\n", " ")

        lines.append(f"  {fn.__name__}({params_str})\n    {first_para}")
    return "\n\n".join(lines)


def _parse_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call> blocks from LLM output."""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            calls.append(parsed)
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                # Handle single quotes
                fixed = match.replace("'", '"')
                parsed = json.loads(fixed)
                calls.append(parsed)
            except json.JSONDecodeError:
                pass
    return calls


def _extract_think(text: str) -> tuple[str, str]:
    """Separate <think>...</think> from the rest."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return thinking, clean


def _execute_tool(fn, arguments: dict) -> str:
    """Execute a tool function with the given arguments."""
    try:
        result = fn(**arguments)
        if asyncio.iscoroutinefunction(fn):
            result = asyncio.run(result)
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
    """Run one agent cycle with text-based tool calling.

    The agent generates text, we parse <tool_call> blocks, execute them,
    feed results back, repeat until no more tool calls or max_turns.

    Returns:
        The agent's final text response.
    """
    import aiohttp

    # Build tool map
    tool_map = {fn.__name__: fn for fn in tools}

    # Build system prompt with tool descriptions
    tool_desc = _build_tool_descriptions(tools)
    full_system = f"""{system_prompt}

## Available Tools
When you want to call a tool, use this exact format (you can make multiple calls):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

After each tool call, you'll receive the result. Then decide your next action.

Here are your tools:
{tool_desc}
"""

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_input},
    ]

    async def _complete(msgs):
        async with aiohttp.ClientSession(timeout=lm.timeout) as session:
            body = {
                "model": lm.model,
                "messages": msgs,
                "stream": False,
                **lm.defaults,
            }
            async with session.post(
                f"{lm.api_base}/v1/chat/completions", json=body
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"LLM error: {data}")
                return data["choices"][0]["message"]["content"]

    final_response = ""

    for turn in range(max_turns):
        # Get LLM response
        response = asyncio.run(_complete(messages))

        # Separate thinking from content
        thinking, clean_response = _extract_think(response)
        if verbose and thinking:
            print(f"  [think] {thinking[:150]}...", file=sys.stderr)

        # Parse tool calls
        tool_calls = _parse_tool_calls(clean_response)

        if not tool_calls:
            # No tool calls — agent is done with this cycle
            # Remove any tool_call tags that might have been malformed
            final_response = re.sub(r'</?tool_call>', '', clean_response).strip()
            if verbose:
                print(f"  [turn {turn+1}] Final response (no tools)", file=sys.stderr)
            break

        # Execute tool calls
        messages.append({"role": "assistant", "content": response})

        tool_results = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})

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

            tool_results.append(f"<tool_response>\n{name}: {result}\n</tool_response>")

        # Feed results back
        messages.append({"role": "user", "content": "\n".join(tool_results)})

    else:
        # Max turns reached
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
