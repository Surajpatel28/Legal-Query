from functools import lru_cache
import logging
import re
from src.database import get_qdrant_client, get_embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.70

def _normalize_query(query: str) -> str:
    """Create a stable cache key for semantically equivalent phrasing."""
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", str(query).lower())
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=1)
def get_cache_vectorstore():
    client = get_qdrant_client()
    embedding = get_embeddings()

    return QdrantVectorStore(
        client=client,
        embedding=embedding,
        collection_name="semantic-cache"
    )


def get_cached_response(query):

    vectorstore = get_cache_vectorstore()
    normalized_query = _normalize_query(query)

    # Search with normalized query and inspect a few nearest neighbors.
    results = vectorstore.similarity_search_with_score(normalized_query, k=3)

    if not results:
        return None

    for doc, score in results:
        cached_normalized_query = str(doc.metadata.get("normalized_query", "")).strip()

        # Exact normalized-key match is a direct cache hit.
        if cached_normalized_query and cached_normalized_query == normalized_query:
            return doc.metadata.get("response") or doc.page_content

        # Otherwise allow high-confidence semantic hit.
        if score >= SIM_THRESHOLD:
            return doc.metadata.get("response") or doc.page_content

    return None


def store_cached_response(query, response):

    vectorstore = get_cache_vectorstore()
    normalized_query = _normalize_query(query)

    doc = Document(
        # Store the query text in vector space for reliable semantic lookup.
        page_content=normalized_query,
        metadata={
            "query": query,
            "normalized_query": normalized_query,
            "response": response,
        }
    )

    logger.debug("STORING RESPONSE IN CACHE")
    vectorstore.add_documents(
        documents=[doc],
        ids=[str(uuid.uuid4())]
    )