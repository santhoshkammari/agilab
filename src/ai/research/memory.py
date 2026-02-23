"""
Generic ChromaDB memory tools for AI agents.
No research-specific logic — just primitives the LLM composes.
"""
import chromadb
import uuid
from typing import Optional

# Module-level client — persistent storage
_client: Optional[chromadb.ClientAPI] = None
_persist_path: str = ".chromadb"


def init(path: str = ".chromadb"):
    """Initialize ChromaDB with a persistence path."""
    global _client, _persist_path
    _persist_path = path
    _client = chromadb.PersistentClient(path=path)
    return _client


def _get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_persist_path)
    return _client


def _get_collection(name: str):
    """Get or create a collection by name."""
    return _get_client().get_or_create_collection(name=name)


def memory_store(collection: str, text: str, metadata: dict = {}, id: str = "") -> str:
    """Store a document in a ChromaDB collection.

    Args:
        collection: Name of the collection (e.g. 'research_data', 'mind', 'searches', 'queue', 'state')
        text: The text content to store
        metadata: Key-value metadata to attach (e.g. {"source_url": "...", "concepts": "RAG,agents"})
        id: Optional document ID. If empty, auto-generates a UUID.

    Returns:
        Confirmation string with the document ID.
    """
    col = _get_collection(collection)
    doc_id = id if id else str(uuid.uuid4())

    # ChromaDB metadata values must be str, int, float, or bool
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            clean_meta[k] = ",".join(str(i) for i in v)
        elif isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)

    col.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[clean_meta] if clean_meta else None
    )
    return f"Stored in '{collection}' (id: {doc_id})"


def memory_search(collection: str, query: str, n: int = 5) -> str:
    """Semantic search across a ChromaDB collection.

    Args:
        collection: Name of the collection to search
        query: The search query (will be embedded and compared semantically)
        n: Maximum number of results to return (default 5)

    Returns:
        Matching documents with their text, metadata, and distance scores.
        Returns 'No results found' if collection is empty or no matches.
    """
    col = _get_collection(collection)

    if col.count() == 0:
        return f"Collection '{collection}' is empty. No results found."

    n = min(n, col.count())
    results = col.query(query_texts=[query], n_results=n)

    if not results['documents'] or not results['documents'][0]:
        return "No results found."

    output = []
    for i, (doc, meta, dist, doc_id) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
        results['ids'][0]
    )):
        # Truncate long docs for display
        preview = doc[:500] + "..." if len(doc) > 500 else doc
        output.append(f"[{i+1}] id={doc_id} (distance={dist:.3f})\n    {preview}\n    metadata={meta}")

    return "\n".join(output)


def memory_exists(collection: str, id: str) -> bool:
    """Check if a document ID exists in a collection.

    Args:
        collection: Name of the collection
        id: The document ID to check

    Returns:
        True if the document exists, False otherwise.
    """
    col = _get_collection(collection)
    try:
        result = col.get(ids=[id])
        return len(result['ids']) > 0
    except Exception:
        return False


def memory_delete(collection: str, id: str) -> str:
    """Delete a document from a collection by ID.

    Args:
        collection: Name of the collection
        id: The document ID to delete

    Returns:
        Confirmation string.
    """
    col = _get_collection(collection)
    col.delete(ids=[id])
    return f"Deleted '{id}' from '{collection}'"


def memory_update(collection: str, id: str, text: str, metadata: dict = {}) -> str:
    """Update an existing document in a collection.

    Args:
        collection: Name of the collection
        id: The document ID to update
        text: New text content
        metadata: New metadata (replaces existing)

    Returns:
        Confirmation string.
    """
    col = _get_collection(collection)

    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            clean_meta[k] = ",".join(str(i) for i in v)
        elif isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)

    col.update(
        ids=[id],
        documents=[text],
        metadatas=[clean_meta] if clean_meta else None
    )
    return f"Updated '{id}' in '{collection}'"
