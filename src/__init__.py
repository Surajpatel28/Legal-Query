from functools import partial
import logging
from src.database import get_vectorstore
from src.retrieval import build_compression_retriever, retrieve
from src.chain import create_rag_chain
from src.caching import get_cached_response, store_cached_response

logger = logging.getLogger(__name__)


def initialize_rag_system():
    try:
        vectorstore = get_vectorstore()
        
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 5}
        )

        compression_retriever = build_compression_retriever(retriever)
        
        retrieve_fn = partial(
            retrieve,
            vectorstore=vectorstore,
            retriever=retriever,
            compression_retriever=compression_retriever
        )

        return create_rag_chain(
            retrieve_fn=retrieve_fn,
            cache_get_fn=get_cached_response,
            cache_store_fn=store_cached_response
        )
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise
