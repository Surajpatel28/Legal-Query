import re
from typing import List
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

reranker = CrossEncoder("BAAI/bge-reranker-base")


def extract_section(query: str):
    try:
        match = re.search(r'\b(?:section|sec|s\.)\s*(\d+)\b', query.lower())
        return match.group(1) if match else None
    except Exception:
        return None

def retrieve(query, vectorstore, retriever) -> List[Document]:
    try:
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
                            match=MatchValue(value=section)
                        )
                    ]
                )
            )
        
            if docs:
                return docs 
        
        docs = retriever.invoke(query)
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )
        return [d for _, d in ranked[:1]]
    except Exception as e:
        print(f"Error during retrieval: {e}")
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
        print(f"Error formatting documents: {e}")
        return "Error formatting documents."

def make_bns_tool(vectorstore, retriever):
    @tool
    def bns_search(query: str) -> str:
        """Search the Bharatiya Nyaya Sanhita (BNS) knowledge base for relevant Indian criminal law sections. Use this first for any Indian law or BNS related question."""
        docs = retrieve(query, vectorstore, retriever)
        return format_docs(docs)
    return bns_search
