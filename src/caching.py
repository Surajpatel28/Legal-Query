from functools import lru_cache
import logging
from src.database import get_qdrant_client, get_embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.85


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

    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return None

    doc, score = results[0]

    # QdrantVectorStore returns cosine similarity directly (higher = more similar)
    if score >= SIM_THRESHOLD:
        return doc.page_content

    return None


def store_cached_response(query, response):

    vectorstore = get_cache_vectorstore()

    doc = Document(
        page_content=response,
        metadata={"query": query}
    )

    logger.debug("STORING RESPONSE IN CACHE")
    vectorstore.add_documents(
        documents=[doc],
        ids=[str(uuid.uuid4())]
    )