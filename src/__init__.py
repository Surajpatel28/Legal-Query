import os 
from langchain_core.runnables import RunnableLambda
from src.database import get_qdrant_client, get_embeddings, get_vectorstore
from src.retrieval import make_bns_tool
from src.chain import create_react_rag_chain
from src.config import TAVILY_API_KEY
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

def initialize_rag_system():
    try:
        client = get_qdrant_client()
        embedding = get_embeddings()
        vectorstore = get_vectorstore(client, embedding)
        
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 5}
        )
        
        bns_tool = make_bns_tool(vectorstore, retriever)

        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        )

        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        web_tool = TavilySearch(max_results=3)
        
        chain = create_react_rag_chain([bns_tool, wikipedia_tool, web_tool])
        return chain
    
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise
