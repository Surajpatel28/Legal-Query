from src.database import get_qdrant_client

client = get_qdrant_client()

client.create_collection(
    collection_name="semantic-cache",
    vectors_config={
        "size": 768,
        "distance": "Cosine"
    }
)

print("semantic-cache collection created")