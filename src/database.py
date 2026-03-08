from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY, COLLECTION_NAME


def get_qdrant_client():
    try:
        client = QdrantClient(
            url=QDRANT_CLOUD_URL,
            api_key=QDRANT_CLOUD_API_KEY,
            prefer_grpc=False
        )
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        raise


def get_embeddings():
    try:
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embedding
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise

 
def get_vectorstore(client, embedding):
    try:
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
        print(f"Error creating vectorstore: {e}")
        raise
