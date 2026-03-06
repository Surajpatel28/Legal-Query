from langchain_core.runnables import RunnableLambda
from src.database import get_qdrant_client, get_embeddings, get_vectorstore
from src.retrieval import retrieve
from src.chain import create_rag_chain_with_memory

def initialize_rag_system():
    try:
        client = get_qdrant_client()
        embedding = get_embeddings()
        vectorstore = get_vectorstore(client, embedding)
        
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 10}
        )
        
        def retrieve_wrapper(query: str):
            return retrieve(query, vectorstore, retriever)
        
        retrieval_chain = RunnableLambda(retrieve_wrapper)
        
        chain = create_rag_chain_with_memory(retrieval_chain)
        
        return chain
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise
