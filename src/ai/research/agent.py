"""
Research Agent — autonomous tool-driven research using ChromaDB as working memory.
All intelligence lives in the system prompt. Tools are generic primitives.

Uses OpenAI-compatible tool calling API (vLLM with --enable-auto-tool-choice).
"""
import sys
import os
import json
import inspect

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NO_THINK = True  # Append /no_think to system prompt (Qwen3: skip <think> block, saves tokens)

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_tools'))

from ai import LM, configure
from research.memory import (
    init as init_memory, _get_client,
    memory_store, memory_search, memory_exists,
    memory_delete, memory_update, memory_count,
    memory_get, memory_get_all
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
# Markdown analysis tools (for synthesis agents)
# ---------------------------------------------------------------------------

def md_get_source_urls() -> str:
    """List all source URLs stored in the research_data collection.

    Returns:
        Numbered list of all source URLs available for analysis.
    """
    items = memory_get_all("research_data")
    if not items:
        return "No sources in research_data."
    lines = [f"[{i+1}] {item['id']} ({item['metadata'].get('char_count', '?')} chars)"
             for i, item in enumerate(items)]
    return "\n".join(lines)


def md_get_content(url: str) -> str:
    """Retrieve the full markdown content for a source URL from research_data.

    Args:
        url: The source URL (document ID in research_data)

    Returns:
        The full markdown content, or error if not found.
    """
    return memory_get("research_data", url)


def md_analyze_structure(url: str) -> str:
    """Analyze the markdown structure of a source: headers, word count, paragraph count, code blocks, lists, tables.

    Args:
        url: The source URL to analyze

    Returns:
        JSON string with structural analysis (headers, word count, etc.)
    """
    from markdown.mrkdwn_analysis import MarkdownAnalyzer
    content = memory_get("research_data", url)
    if content.startswith("No document found") or content.startswith("Error"):
        return content
    try:
        analyzer = MarkdownAnalyzer.from_string(content)
        stats = analyzer.analyse()
        headers = analyzer.identify_headers().get("Header", [])
        return json.dumps({
            'url': url,
            'stats': stats,
            'headers': [{'level': h['level'], 'text': h['text']} for h in headers],
        }, indent=2)
    except Exception as e:
        return f"Analysis error for {url}: {e}"


def md_extract_headers(url: str) -> str:
    """Extract all headers from a source's markdown content.

    Args:
        url: The source URL to extract headers from

    Returns:
        List of headers with their levels.
    """
    from markdown.mrkdwn_analysis import MarkdownAnalyzer
    content = memory_get("research_data", url)
    if content.startswith("No document found") or content.startswith("Error"):
        return content
    try:
        analyzer = MarkdownAnalyzer.from_string(content)
        headers = analyzer.identify_headers().get("Header", [])
        lines = [f"H{h['level']}: {h['text']}" for h in headers]
        return "\n".join(lines) if lines else "No headers found."
    except Exception as e:
        return f"Error: {e}"


def md_extract_paragraphs(url: str, max_paragraphs: int = 10) -> str:
    """Extract paragraphs from a source's markdown content.

    Args:
        url: The source URL to extract paragraphs from
        max_paragraphs: Maximum number of paragraphs to return (default 10)

    Returns:
        Numbered paragraphs from the source.
    """
    from markdown.mrkdwn_analysis import MarkdownAnalyzer
    content = memory_get("research_data", url)
    if content.startswith("No document found") or content.startswith("Error"):
        return content
    try:
        analyzer = MarkdownAnalyzer.from_string(content)
        paragraphs = analyzer.identify_paragraphs().get("Paragraph", [])
        lines = [f"[{i+1}] {p[:500]}" for i, p in enumerate(paragraphs[:max_paragraphs])]
        return "\n\n".join(lines) if lines else "No paragraphs found."
    except Exception as e:
        return f"Error: {e}"


def md_extract_links(url: str) -> str:
    """Extract all links from a source's markdown content.

    Args:
        url: The source URL to extract links from

    Returns:
        List of links found in the markdown.
    """
    from markdown.mrkdwn_analysis import MarkdownAnalyzer
    content = memory_get("research_data", url)
    if content.startswith("No document found") or content.startswith("Error"):
        return content
    try:
        analyzer = MarkdownAnalyzer.from_string(content)
        links = analyzer.identify_links()
        lines = []
        for l in links.get("Text link", []):
            lines.append(f"[{l.get('text', '')}]({l.get('url', '')})")
        for img in links.get("Image link", []):
            lines.append(f"![{img.get('alt_text', '')}]({img.get('url', '')})")
        return "\n".join(lines) if lines else "No links found."
    except Exception as e:
        return f"Error: {e}"


def md_extract_code_blocks(url: str) -> str:
    """Extract code blocks from a source's markdown content.

    Args:
        url: The source URL to extract code blocks from

    Returns:
        Code blocks with their language tags.
    """
    from markdown.mrkdwn_analysis import MarkdownAnalyzer
    content = memory_get("research_data", url)
    if content.startswith("No document found") or content.startswith("Error"):
        return content
    try:
        analyzer = MarkdownAnalyzer.from_string(content)
        blocks = analyzer.identify_code_blocks().get("Code block", [])
        lines = []
        for b in blocks:
            lang = b.get('language') or 'unknown'
            preview = b['content'][:300]
            lines.append(f"```{lang}\n{preview}\n```")
        return "\n\n".join(lines) if lines else "No code blocks found."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# File tools — for answer.md and progress.md
# ---------------------------------------------------------------------------

_answer_path = "answer.md"
_progress_path = "progress.md"


def write_answer(content: str) -> str:
    """Write the final research answer to answer.md. Use this to save your final deliverable.

    Args:
        content: The full answer in markdown format.

    Returns:
        Confirmation message.
    """
    with open(_answer_path, 'w') as f:
        f.write(content)
    return f"Answer written to {_answer_path} ({len(content)} chars)"


def read_answer() -> str:
    """Read the current answer.md file.

    Returns:
        Contents of answer.md, or message if not found.
    """
    try:
        with open(_answer_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "No answer file found yet."


def write_progress(content: str) -> str:
    """Write/overwrite the progress file.

    Args:
        content: The full progress content. Example:
            # TASK: find 5 unique blogs on genai
            [3/5]

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

    sys_content = (system_prompt + "\n/no_think") if NO_THINK else system_prompt
    messages = [
        {"role": "system", "content": sys_content},
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

RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent. Each invocation = ONE short cycle. You are STATELESS -- memory is ONLY in ChromaDB and progress.md.

## Collections
- research_data: fetched page content (auto-stored by web_fetch)
- mind: your observations per source (THIS is what counts toward the target)
- searches: query log (prevents repeats)
- queue: leads to follow

## CYCLE (follow EXACTLY, keep it under ~12 tool calls)

1. READ STATE (2 calls):
   read_progress() and memory_count("mind")
   If mind count >= target number from the task, respond "EXPLORATION_COMPLETE" and STOP.

2. SEARCH (1 call):
   web_search(query) -- vary your query each cycle, never repeat.

3. FETCH (1-2 calls max):
   Pick 1-2 best URLs. web_fetch(url) for each.

4. ABSORB (1-2 calls):
   For each fetched URL, store ONE observation:
   memory_store("mind", "Source: X | IDEA: ... | ANGLE: ... | CONCEPTS: a,b,c",
                {"source_url": url, "type": "absorption"})

5. UPDATE PROGRESS and STOP (2 calls then stop):
   Call memory_count("mind") to get the real count.
   Call write_progress() with exactly this format:
   # TASK: <the user's original task>
   [COUNT/TARGET]
   Then respond with text: "Cycle done. COUNT/TARGET." and STOP.

## Rules
- Max 1 web_search, 2 web_fetch per cycle. NO MORE.
- Always end with a text response. Never run out of turns silently.
- Do NOT synthesize or summarize findings. Just collect data. Synthesis happens separately.
- If stuck, respond "EXHAUSTED"."""


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
        answer_path: str = "answer.md",
        max_cycles: int = 200,
        max_turns_per_cycle: int = 20,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        self.max_cycles = max_cycles
        self.max_turns_per_cycle = max_turns_per_cycle
        self.verbose = verbose

        global _progress_path, _answer_path
        _progress_path = progress_path
        _answer_path = answer_path

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
            memory_delete, memory_update, memory_count,
            web_search, web_fetch,
            write_progress, read_progress,
        ]

    @staticmethod
    def _parse_target(task: str) -> int:
        """Extract the target number from the task string (e.g. 'find 5 blogs' -> 5)."""
        import re
        m = re.search(r'(\d+)', task)
        return int(m.group(1)) if m else 0

    def reset(self):
        """Clear all ChromaDB collections and progress file for a fresh start."""
        client = _get_client()
        for name in ["research_data", "mind", "searches", "queue", "synthesis"]:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        for f in [_progress_path, _answer_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        print("Reset: cleared all collections and progress.")

    # ------------------------------------------------------------------
    # Synthesis Pipeline (tool-calling agents)
    # ------------------------------------------------------------------

    def _synthesis_tools(self) -> list:
        """Tools available to synthesis agents: markdown analysis + memory + file output."""
        return [
            md_get_source_urls, md_get_content,
            md_analyze_structure, md_extract_headers,
            md_extract_paragraphs, md_extract_links,
            md_extract_code_blocks,
            memory_search, memory_store, memory_get, memory_count,
            write_answer, read_answer,
        ]

    def _run_analyst_for_source(self, task: str, url: str, index: int, total: int) -> str:
        """Run one analyst agent for a single source URL. Thread-safe."""
        analyst_prompt = f"""You are research analyst #{index}/{total}. Your job: deeply analyze ONE source.

## Your assigned source
URL: {url}

## Your tools
- md_get_content(url): Get full markdown content
- md_analyze_structure(url): Get structural stats (headers, word count, etc.)
- md_extract_headers(url): Extract all headers
- md_extract_paragraphs(url): Extract key paragraphs
- md_extract_links(url): Extract all links
- md_extract_code_blocks(url): Extract code blocks
- memory_store(collection, text, metadata, id): Store your analysis

## Instructions
1. Use markdown tools to deeply understand your assigned source
2. Produce a structured summary:
   - Source URL
   - Key arguments / main points
   - Unique angle (what makes this source different)
   - Evidence / data / examples
   - Key concepts
3. Store your analysis: memory_store("synthesis", <your full summary>, {{"phase": "analyst", "source_url": "{url}"}})
4. Respond with your summary text."""

        tools = self._synthesis_tools()
        return run_agent_cycle(
            lm=self.lm,
            system_prompt=analyst_prompt,
            user_input=f"Analyze this source for the task: {task}",
            tools=tools,
            max_turns=self.max_turns_per_cycle,
            verbose=self.verbose,
        )

    def _synthesize(self, task: str) -> str:
        """Run the synthesis pipeline:
        Parallel Analysts (1 per source) → Critic (with feedback loop) → Writer.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\n{'='*60}")
        print("PHASE 2: SYNTHESIS")
        print(f"{'='*60}")

        tools = self._synthesis_tools()

        # Get all source URLs
        source_items = memory_get_all("research_data")
        source_urls = [item['id'] for item in source_items]

        if not source_urls:
            return "No research data found to synthesize."

        total = len(source_urls)
        print(f"\n  Sources to analyze: {total}")

        # --- PARALLEL ANALYSTS (1 per source) ---
        print(f"\n  [Analysts] Launching {total} parallel analyst agents...")

        analyst_results = {}
        with ThreadPoolExecutor(max_workers=min(total, 8)) as executor:
            futures = {
                executor.submit(self._run_analyst_for_source, task, url, i+1, total): url
                for i, url in enumerate(source_urls)
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    result = future.result()
                    analyst_results[url] = result
                    print(f"  [Analyst] Done: {url[:60]}... ({len(result)} chars)")
                except Exception as e:
                    print(f"  [Analyst] FAILED: {url[:60]}... — {e}")
                    analyst_results[url] = f"Analysis failed: {e}"

        print(f"  [Analysts] All {len(analyst_results)}/{total} complete")

        # --- CRITIC (max 2 rounds) ---
        max_critic_rounds = 2
        for critic_round in range(1, max_critic_rounds + 1):
            print(f"\n  [Critic] Round {critic_round}/{max_critic_rounds}...")

            critic_prompt = """You are a research critic. Multiple analyst agents have each analyzed one source. Their summaries are stored in ChromaDB.

## Your tools
- md_get_source_urls(): List all sources
- md_get_content(url), md_analyze_structure(url), md_extract_headers(url), md_extract_paragraphs(url), md_extract_links(url): Inspect raw sources
- memory_search("synthesis", query, n): Retrieve analyst summaries (search with relevant terms)
- memory_search("mind", "research", n=50): See exploration observations
- memory_store("synthesis", ..., {"phase": "critic"}): Store your critique

## Instructions
1. Search memory for all analyst summaries: memory_search("synthesis", "analyst summary", n=50)
2. Cross-check against raw sources using markdown tools where needed
3. Identify: themes, gaps, contradictions, source ranking by relevance
4. Flag weak or duplicate analyses

If analyses are comprehensive: respond starting with "APPROVED" followed by your thematic analysis.
If they need revision: respond starting with "FEEDBACK:" with specific requests per source.

Store your critique: memory_store("synthesis", <your critique>, {"phase": "critic"})"""

            critic_output = run_agent_cycle(
                lm=self.lm,
                system_prompt=critic_prompt,
                user_input=f"Critique the analyst summaries for task: {task}",
                tools=tools,
                max_turns=self.max_turns_per_cycle,
                verbose=self.verbose,
            )
            print(f"  [Critic] Done ({len(critic_output)} chars)")

            if "APPROVED" in critic_output:
                print("  [Critic] APPROVED")
                break

            # Feedback round — re-run analysts that need revision (parallel)
            print("  [Analysts] Revising based on critic feedback...")

            revision_prompt = """You are a research analyst revising your work based on critic feedback.

## Your tools
- md_get_content(url), md_analyze_structure(url), md_extract_headers(url), md_extract_paragraphs(url), md_extract_links(url), md_extract_code_blocks(url)
- memory_search("synthesis", "critic"): Get the critic's feedback
- memory_search("synthesis", "analyst"): See previous analyses
- memory_store("synthesis", ..., {"phase": "analyst"}): Store revised analysis

## Instructions
1. Read the critic's feedback from memory
2. Use markdown tools to re-examine sources where the critic found gaps
3. Produce revised summaries addressing all feedback
4. Store and respond with the revised analysis."""

            analyst_output = run_agent_cycle(
                lm=self.lm,
                system_prompt=revision_prompt,
                user_input=f"Revise the analyses for task: {task}",
                tools=tools,
                max_turns=self.max_turns_per_cycle,
                verbose=self.verbose,
            )
            print(f"  [Analyst revision] Done ({len(analyst_output)} chars)")

        # --- WRITER ---
        print(f"\n  [Writer] Producing final answer...")

        writer_prompt = """You are a research writer producing the final deliverable. Multiple analysts have each analyzed one source, and a critic has reviewed their work. Everything is in ChromaDB.

## Your tools
- memory_search("synthesis", "analyst", n=50): Get all analyst summaries
- memory_search("synthesis", "critic"): Get critic's thematic analysis
- memory_search("mind", "research", n=50): See exploration observations
- md_get_source_urls(), md_get_content(url), md_extract_paragraphs(url): Inspect sources directly for quotes
- write_answer(content): Save the final answer to answer.md
- read_answer(): Read the current answer.md

## Instructions
1. Retrieve ALL analyst summaries and the critic's analysis from memory
2. Optionally inspect specific sources for quotes or details
3. Produce a comprehensive, well-structured answer:
   - Cohesive narrative organized by themes
   - Cite sources with URLs
   - Include key evidence and unique perspectives from each source
   - End with synthesis/conclusion
   - Use markdown formatting
4. IMPORTANT: Call write_answer(content) to save your final answer to answer.md

Respond with a brief confirmation after saving."""

        final_answer = run_agent_cycle(
            lm=self.lm,
            system_prompt=writer_prompt,
            user_input=task,
            tools=tools,
            max_turns=self.max_turns_per_cycle,
            verbose=self.verbose,
        )

        # Prefer answer.md content (complete) over LLM text response (may be truncated)
        try:
            with open(_answer_path, 'r') as f:
                saved = f.read().strip()
            if saved:
                final_answer = saved
        except FileNotFoundError:
            # Writer didn't call write_answer — save its response ourselves
            if final_answer.strip():
                with open(_answer_path, 'w') as f:
                    f.write(final_answer)

        print(f"  [Writer] Done ({len(final_answer)} chars)")
        print(f"  Answer saved to: {_answer_path}")

        print(f"\n{'='*60}")
        print("SYNTHESIS COMPLETE")
        print(f"{'='*60}")

        return final_answer

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, task: str, fresh: bool = False):
        """Run the research agent: Phase 1 (exploration) then Phase 2 (synthesis)."""
        if fresh:
            self.reset()

        target = self._parse_target(task)

        print(f"{'='*60}")
        print(f"RESEARCH AGENT")
        print(f"Task: {task}")
        print(f"Target: {target or '?'} | Max cycles: {self.max_cycles}")
        print(f"{'='*60}")
        print(f"\nPHASE 1: EXPLORATION")
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

            # Check completion: trust the data, not the LLM's words
            mind_count = memory_count("mind")
            if "EXPLORATION_COMPLETE" in response or "DONE" in response or (target and mind_count >= target):
                print(f"\n{'='*60}")
                print(f"EXPLORATION done after {cycle} cycles ({mind_count} observations)")
                print(f"{'='*60}")
                # Phase 2: Synthesis
                return self._synthesize(task)

            if "EXHAUSTED" in response:
                print(f"\n{'='*60}")
                print(f"EXHAUSTED after {cycle} cycles ({mind_count} observations)")
                print(f"{'='*60}")
                # Still synthesize whatever we have
                if mind_count > 0:
                    return self._synthesize(task)
                return response

        print(f"\n{'='*60}")
        print(f"Reached max cycles ({self.max_cycles})")
        print(f"{'='*60}")
        # Synthesize whatever we collected
        mind_count = memory_count("mind")
        if mind_count > 0:
            return self._synthesize(task)
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
    parser.add_argument("--fresh", action="store_true", help="Clear all data and start fresh")
    args = parser.parse_args()

    agent = ResearchAgent(
        model=args.model,
        api_base=args.api_base,
        max_cycles=args.max_cycles,
        chromadb_path=args.chromadb_path,
        verbose=args.verbose,
    )
    agent.run(args.task, fresh=args.fresh)
