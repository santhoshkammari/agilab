# Research Agent System — Design Document

## Problem Statement

Build an autonomous research agent that can execute open-ended research tasks like:
- "Find 15 unique blogs on GenAI with diverse perspectives"
- "Search for 50 analog watches under 5K INR from Indian shopping sites"
- "Collect 20 academic papers on AI safety with different methodologies"

The agent must be **purely tool-driven** — no hardcoded research logic, no custom modules. All intelligence lives in the system prompt + tool interactions. The same tools work for any research task.

---

## Core Philosophy

1. **Stateless agent, stateful memory** — agent wakes up fresh each invocation, reads state from ChromaDB, does one cycle, writes state back, dies
2. **ChromaDB as RAM** — collections serve as working memory, like how humans use notepads
3. **Semantic search as concept clustering** — no k-means, no code; vector similarity IS the clustering
4. **Markdown agent for content reading** — existing MCP markdown tools handle reading/analyzing stored content
5. **Generic tools only** — no task-specific tools; LLM composes workflow from primitives
6. **tqdm-style progress** — a progress.md file the agent reads/overwrites each cycle for self-regulation

---

## Architecture

```
OUTER LOOP (simple Python for/while)
│
├──→ Agent wakes up (stateless, fresh context)
│    │
│    ├→ Reads progress.md (where am I?)
│    ├→ Reads ChromaDB collections (what do I know?)
│    ├→ Decides next action (from queue, gaps, or fresh query)
│    ├→ Executes ONE meaningful cycle
│    │   ├→ Search web
│    │   ├→ Fetch URLs → auto-stored in research_data
│    │   ├→ Absorb ideas (via markdown agent + semantic probing)
│    │   ├→ Extract leads → queue
│    │   ├→ Check uniqueness via semantic search on mind
│    │   └→ Update progress
│    ├→ Writes updated state back to ChromaDB + progress.md
│    └→ Returns "DONE" / "EXHAUSTED" / "Completed cycle. X/Y."
│
└──→ Loop checks response, continues or stops
```

---

## The 5 ChromaDB Collections

### 1. `research_data` — The Filing Cabinet
Raw fetched content. The web_fetch tool auto-stores here.

| Field | Value |
|-------|-------|
| ID | URL |
| Text | Full markdown content |
| Metadata | `{title, fetched_at, source_query}` |

**Used for**: Dedup (check if URL already fetched), content analysis via markdown agent.

### 2. `mind` — The Understanding
Absorbed observations from each piece of content. This is what the agent "learned".

| Field | Value |
|-------|-------|
| ID | Auto-generated |
| Text | Observation (main idea, author stance, concepts, uniqueness notes) |
| Metadata | `{source_url, concepts[], author_stance, type: "absorption"}` |

**Used for**: Concept clustering (semantic search), uniqueness checking, gap detection.

Example entry:
```
Text: "Blog by Lilian Weng on RAG. Deep academic treatment, references
      12 papers. Covers retrieval, chunking, embedding models. Argues
      chunk size matters more than embedding choice. Links to FAISS docs
      and LlamaIndex blog."
Metadata: {
  source_url: "https://lilianweng.github.io/...",
  concepts: ["RAG", "retrieval", "chunking", "embeddings", "FAISS"],
  author_stance: "academic, evidence-based",
  type: "absorption"
}
```

### 3. `searches` — The Search Log
Every web search query the agent has run and what it yielded.

| Field | Value |
|-------|-------|
| ID | Auto-generated |
| Text | The search query string |
| Metadata | `{results_count, new_urls_found, already_seen, useful: bool}` |

**Used for**: Avoiding duplicate queries, detecting diminishing returns, strategy adjustment.

### 4. `queue` — The "To Explore" Sticky Notes
Leads that emerged from absorbed content — linked URLs, query ideas, follow-up concepts.

| Field | Value |
|-------|-------|
| ID | Auto-generated |
| Text | URL or query idea |
| Metadata | `{type: "url_candidate"|"query_candidate", emerged_from, reason}` |

**Used for**: Agent picks next action from here before generating fresh queries.

Example entries:
```
Text: "https://jayalammar.github.io/illustrated-transformer/"
Metadata: {type: "url_candidate", emerged_from: "https://blog-x.com",
           reason: "referenced as foundational reading"}

Text: "full fine-tuning vs LoRA for models under 1B parameters"
Metadata: {type: "query_candidate", emerged_from: "https://blog-y.com",
           reason: "author's contrarian claim worth verifying"}
```

### 5. `state` — The Progress Card
Single document that tracks overall progress.

| Field | Value |
|-------|-------|
| ID | "progress" |
| Text | JSON with task status |
| Metadata | `{target, completed, strategy}` |

Also mirrored as `progress.md` file for tqdm-style display.

---

## The Tools (9 total, all generic)

### ChromaDB Tools
```python
memory_store(collection: str, text: str, metadata: dict) -> str
    # Stores a document in the named collection
    # Returns: "Stored in {collection} (id: {id})"

memory_search(collection: str, query: str, n: int = 5) -> str
    # Semantic search across a collection
    # Returns: matching documents with text + metadata

memory_exists(collection: str, id: str) -> bool
    # Check if a document ID exists in collection
    # Returns: true/false

memory_delete(collection: str, id: str) -> str
    # Remove a document from collection
    # Returns: "Deleted from {collection}"

memory_update(collection: str, id: str, text: str, metadata: dict) -> str
    # Update an existing document
    # Returns: "Updated in {collection}"
```

### Web Tools
```python
web_search(query: str) -> str
    # Runs web search, returns URLs + snippets
    # Does NOT auto-store anything (agent decides what to pursue)

web_fetch(url: str) -> str
    # Fetches URL, converts to markdown, AUTO-STORES in research_data collection
    # Returns: "Added to research_data: {url} ({char_count} chars)"
    # Agent NEVER sees the full markdown in its context
```

### File Tools
```python
write_file(path: str, content: str) -> str
    # Overwrites file with content (used for progress.md)
    # Returns: "Written to {path}"

read_file(path: str) -> str
    # Reads file content (used for progress.md)
    # Returns: file contents
```

### Content Analysis
For extracting links, understanding author's perspective, analyzing blog content:
**Use existing MCP markdown tools / markdown agent** — not a custom tool.
The agent invokes the markdown reading tools to analyze content stored in research_data.

---

## The Agent Cycle (One Invocation)

```
1. READ STATE
   ├→ read_file("progress.md")
   ├→ If target reached → respond "DONE", stop
   └→ If target not reached → continue

2. DECIDE NEXT ACTION
   ├→ memory_search("queue", "pending leads", n=5)
   │   → If queue has leads → pick best one
   │   → If queue empty → go to step 2b
   │
   ├→ 2b: DETECT CONCEPT GAPS
   │   ├→ memory_search("mind", "concept area A", n=5) → covered?
   │   ├→ memory_search("mind", "concept area B", n=5) → gap?
   │   └→ Generate query targeting biggest gap
   │
   └→ Before running query:
       └→ memory_search("searches", query, n=3)
           → If very similar query already run → modify or skip

3. EXECUTE
   ├→ web_search(query)
   │   → Get URLs + snippets
   │
   ├→ For each promising URL:
   │   ├→ memory_exists("research_data", url)
   │   │   → true: skip (already fetched)
   │   │   → false: continue
   │   └→ web_fetch(url) → auto-stored in research_data
   │
   └→ Log query to searches:
       └→ memory_store("searches", query, {results_count, new_found, already_seen})

4. ABSORB (for each newly fetched URL)
   ├→ Use markdown agent / MCP tools to read the stored content:
   │   → What is the main idea / thesis?
   │   → What is the author's unique perspective?
   │   → What links / references do they share?
   │   → What concepts does this content cover?
   │
   ├→ Store observations:
   │   └→ memory_store("mind", observation_text, {source_url, concepts[], author_stance})
   │
   └→ Extract new leads:
       └→ memory_store("queue", lead, {type, emerged_from, reason})
           → Linked URLs mentioned by author
           → Query ideas from new concepts discovered
           → Follow-ups on contrarian/unique claims

5. ASSESS UNIQUENESS (semantic clustering)
   ├→ memory_search("mind", "concepts from this blog", n=10)
   │   → High similarity to many existing entries → NOT unique, don't count
   │   → Low similarity / new concept cluster → UNIQUE, count it
   │
   └→ The semantic distance IS the uniqueness measure
       No code needed. Vector DB does the clustering.

6. UPDATE PROGRESS
   ├→ write_file("progress.md", updated_tqdm_display)
   │   Format:
   │   [██████░░░░░░░░░░░░░░] 6/15
   │   Strategy: following linked leads from absorbed blogs
   │   Last: "Agent Architectures blog" ✓ unique
   │   Gaps: AI safety, multimodal generation
   │   Queue: 5 leads pending
   │
   ├→ memory_update("state", "progress", updated_json, {target, completed})
   └→ Respond: "Completed cycle. 6/15." (or "DONE" or "EXHAUSTED")
```

---

## The Outer Loop (Python)

```python
import ai

# Configure LM
lm = ai.LM(model="...", api_base="...", temperature=0.7)
ai.configure(lm)

# Define task
task = "Find 15 unique blogs on GenAI with diverse perspectives"

# Build agent with generic tools
researcher = ai.Predict(
    system=RESEARCH_SYSTEM_PROMPT,  # all logic lives here
    tools=[
        memory_store, memory_search, memory_exists,
        memory_delete, memory_update,
        web_search, web_fetch,
        write_file, read_file
    ],
    max_iterations=30  # per invocation (not total)
)

# Simple outer loop
for cycle in range(200):  # safety cap
    result = researcher(input=task)

    if "DONE" in result.text or "EXHAUSTED" in result.text:
        print(f"Finished after {cycle + 1} cycles")
        break

    researcher.clear_history()  # stateless — wipe context each cycle
```

---

## How Query Evolution Works (No Hardcoding)

The agent generates new queries through 3 mechanisms:

### 1. Queue-driven (following leads)
Content absorption extracts links and ideas from blogs. These go into the queue collection. Next invocation picks from queue first.

```
Blog mentions "see also: Jay Alammar's illustrated guides"
  → queue: {text: "Jay Alammar illustrated guide AI", type: "url_candidate"}
```

### 2. Gap-driven (concept clustering)
Agent probes mind collection across concept areas. Wherever semantic search returns few/no results = gap. Agent generates query targeting that gap.

```
memory_search("mind", "AI safety alignment") → 0 results
  → Agent generates: "AI safety alignment blog tutorial 2024"
```

### 3. Absorption-driven (concept stealing)
While absorbing a blog, the agent notices new concepts/ideas it hasn't seen before. These become query candidates.

```
Blog argues "LoRA is overrated for small models"
  → Agent has no entries about this debate
  → queue: {text: "LoRA vs full fine-tuning small models comparison", type: "query_candidate"}
```

### Staleness Detection
Agent reads searches collection. If last several queries all had high `already_seen` counts, it knows broad queries are exhausted and must go specific/niche. This happens naturally from reading its own history — no hardcoded threshold.

---

## How Uniqueness Works (Semantic Clustering)

No clustering algorithms. No code. Pure semantic search.

```
Agent absorbs Blog X about "RAG with knowledge graphs"
  → memory_search("mind", "RAG knowledge graphs", n=10)

  Case 1: 0-1 results → NEW CONCEPT CLUSTER → unique ✓
  Case 2: 5+ results, all similar → COVERED CLUSTER → not unique ✗
  Case 3: 2-3 results, but Blog X has a different angle
          → Agent reads the matches and judges → its call
```

The agent decides. The semantic similarity gives it the signal. The LLM interprets the signal. No thresholds in code.

---

## Applicable Examples

### Example 1: GenAI Blogs
```
Task: "Find 15 unique blogs on GenAI with diverse perspectives"
Concept probes: RAG, fine-tuning, agents, safety, multimodal, prompting, evaluation, deployment
Uniqueness: different author perspectives, different concept clusters
Queue feeds: linked blogs, referenced tutorials, mentioned frameworks
```

### Example 2: Analog Watches (Shopping)
```
Task: "Find 50 analog watches under 5000 INR from Indian online stores"
Concept probes: brand names, price ranges, features (water-resistant, leather strap)
Uniqueness: different products (not same watch from different sellers)
Queue feeds: "see also" products, brand pages, comparison articles
Collections adapt: research_data stores product pages, mind stores product details
```

### Example 3: Academic Papers
```
Task: "Collect 20 papers on AI safety with different methodologies"
Concept probes: RLHF, constitutional AI, interpretability, alignment, red-teaming
Uniqueness: different methodologies, different research groups
Queue feeds: cited papers, related work sections, author's other papers
```

**Same 5 collections. Same 9 tools. Same outer loop. Different system prompt.**

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent per invocation | Stateless | Prevents context drift on long tasks |
| Outer loop | Simple Python for loop | Dumb loop, smart agent |
| Query generation | Agent decides (from queue/gaps/absorption) | No hardcoded query templates |
| Uniqueness check | Semantic search on mind collection | Vector distance = concept distance |
| Content analysis | Markdown agent (MCP tools) | Content never enters agent context |
| Progress tracking | progress.md file (tqdm-style overwrite) | Self-regulation + human visibility |
| Tool design | All generic | Same tools for blogs, shopping, papers |
| Queue priority | Agent decides | No hardcoded priority rules |
| Concept gap probing | Agent decides how many probes | No limits on exploration |
| Stop condition | Agent says "DONE" or "EXHAUSTED" | Agent judges when task is complete |

---

## What's Needed to Build This

1. **ChromaDB tool functions** (5 functions wrapping chromadb Python client)
2. **web_search tool** (wrapping existing MCP web search)
3. **web_fetch tool** (wrapping existing MCP web fetch + auto-store to ChromaDB)
4. **The system prompt** (all research logic encoded as instructions)
5. **The outer loop** (5 lines of Python using existing ai.py Predict)

No new modules. No new classes. No frameworks. Just tools + prompt + loop.

---

## Open Questions for Implementation

1. **ChromaDB embedding model** — use default (all-MiniLM-L6-v2) or something better for concept clustering?
2. **Chunk size for research_data** — long blogs need chunking before storing. ChromaDB has limits. Handle in web_fetch tool?
3. **Rate limiting** — web_search and web_fetch may hit rate limits. Handle in tool functions with simple retry?
4. **Collection initialization** — create collections on first run or require explicit setup?
5. **Multiple concurrent tasks** — namespace collections per task? e.g., `genai_research_data`, `genai_mind`?

---

## Sources & References

- [Memory in the Age of AI Agents (arXiv 2512.13564)](https://arxiv.org/abs/2512.13564) — taxonomy of agent memory types
- [ChromaDB MCP Server](https://github.com/djm81/chroma_mcp_server) — existing MCP integration for ChromaDB
- [CrewAI Memory System](https://docs.crewai.com/en/concepts/memory) — short/long/entity memory patterns
- [Agentic Metacognition (arXiv 2509.19783)](https://arxiv.org/html/2509.19783) — self-aware agent design
- [Anthropic: Measuring Agent Autonomy](https://www.anthropic.com/research/measuring-agent-autonomy)
- [OpenAI: Practical Guide to Building Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
