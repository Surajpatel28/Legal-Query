import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_CLOUD_URL = os.getenv('QDRANT_CLOUD_URL')
QDRANT_CLOUD_API_KEY = os.getenv('QDRANT_CLOUD_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
GROQ_MODEL_NAME = os.getenv('GROQ_MODEL_NAME')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
