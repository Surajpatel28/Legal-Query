import re
import logging
from functools import lru_cache
from typing import List
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

logger = logging.getLogger(__name__)
MIN_RELEVANCE_SCORE = 0.35
FALLBACK_TOP_K = 2

# Pre-compile regex patterns once at module level
_RE_SECTION_KEYWORD = re.compile(r'\b(?:section|sec|s\.)\s*(\d{1,4})\b', re.IGNORECASE)
_RE_SECTION_BNS     = re.compile(r'\b(\d{1,4})\s+(?:of\s+)?bns\b', re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_compressor():
    """Lazy-load FlashrankRerank once and cache it."""
    return FlashrankRerank(top_n=3)


def build_compression_retriever(retriever):
    return ContextualCompressionRetriever(
        base_compressor=_get_compressor(),
        base_retriever=retriever
    )


def extract_section(query: str):
    """Extract a BNS section number only when explicitly referenced."""
    q = query.lower()

    # Match "section 103", "sec 103", "s. 103", "s.103"
    match = _RE_SECTION_KEYWORD.search(q)
    if match:
        return match.group(1)

    # Match "103 of BNS", "103 bns"
    match = _RE_SECTION_BNS.search(q)
    if match:
        return match.group(1)

    return None

def retrieve(query, vectorstore, retriever, compression_retriever=None) -> List[Document]:
    try:
        compression_retriever = compression_retriever or build_compression_retriever(retriever)

        if isinstance(query, dict):
            query = query.get('query', str(query))
        
        query = str(query)
        
        section = extract_section(query)

        if section:
            docs = vectorstore.similarity_search(
                query,
                k=1,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.section",
                            match=MatchValue(value=str(section))
                        )
                    ]
                )
            )
            if docs and str(docs[0].metadata.get('section')) == str(section):
                return docs
            return []

        # Retrieve scored candidates first, then keep only relevant evidence.
        scored_docs = vectorstore.similarity_search_with_relevance_scores(query, k=8)
        filtered_docs = [doc for doc, score in scored_docs if score >= MIN_RELEVANCE_SCORE]

        # If strict filtering removes everything, keep a tiny fallback context.
        if not filtered_docs and scored_docs:
            filtered_docs = [doc for doc, _ in scored_docs[:FALLBACK_TOP_K]]

        if not filtered_docs:
            return []

        # Rerank within filtered candidates to reduce unrelated sections in final context.
        reranked_docs = _get_compressor().compress_documents(filtered_docs, query)
        return reranked_docs if reranked_docs else filtered_docs[:FALLBACK_TOP_K]
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []


def format_docs(docs: List[Document]) -> str:
    try:
        if not docs:
            return "No relevant sections found."
        
        context_parts = []
        for doc in docs:
            section = doc.metadata.get('section', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"BNS Section {section}:\n{content}")
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return "Error formatting documents."