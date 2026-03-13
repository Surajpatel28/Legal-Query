import logging
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY, COLLECTION_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client():
    try:
        client = QdrantClient(
            url=QDRANT_CLOUD_URL,
            api_key=QDRANT_CLOUD_API_KEY,
            prefer_grpc=True  # gRPC is faster than HTTP/REST
        )
        return client
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        raise


@lru_cache(maxsize=1)
def get_embeddings():
    try:
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}  # required for correct cosine similarity
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embedding
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise


@lru_cache(maxsize=1)
def get_vectorstore():
    """Cached zero-arg wrapper — avoids reconstructing vectorstore on every call."""
    try:
        client = get_qdrant_client()
        embedding = get_embeddings()
        vectorstore = QdrantVectorStore(
            embedding=embedding,
            client=client,
            collection_name=COLLECTION_NAME
        )

        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.section",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise


def get_vectorstore_with_deps(client=None, embedding=None):
    """Non-cached variant used by the ingestion script with injected deps."""
    try:
        client = client or get_qdrant_client()
        embedding = embedding or get_embeddings()
        return QdrantVectorStore(
            embedding=embedding,
            client=client,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise
