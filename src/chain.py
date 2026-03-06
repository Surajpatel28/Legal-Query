from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.config import GROQ_MODEL_NAME, GROQ_API_KEY
from src.retrieval import format_docs

def get_model():
    try:
        model = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY
        )
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise


def get_prompt():
    prompt = PromptTemplate(
        template="""
You are a helpful legal information assistant. Answer questions about Indian law based on the Bharatiya Nyaya Sanhita (BNS).

Previous conversation:
{chat_history}

Rules:
- Use ONLY the information provided below
- If the information is not in the provided context, say so
- Explain in simple, clear language that anyone can understand
- Use conversation history to understand follow-up questions

Legal sections:
{context}

User question: {query}

Provide your answer in this format:

Section: [section number and title]

What it means:
[explain in simple words what this law is about]

Key points:
- [important point 1]
- [important point 2]
- [additional points if relevant]
""",
        input_variables=["context", "query", "chat_history"]
    )
    return prompt


def create_rag_chain(retrieval_chain):
    try:
        model = get_model()
        prompt = get_prompt()
        parser = StrOutputParser()
        
        chain = (
            {
                'context': retrieval_chain | format_docs,
                'query': RunnablePassthrough()
            } | prompt | model | parser
        )
        
        return chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise

def create_rag_chain_with_memory(retrieval_chain):
    try:
        model = get_model()
        prompt = get_prompt()
        parser = StrOutputParser()

        chain = (
            {
                'context': retrieval_chain | format_docs,
                'query': lambda x: x['query'] if isinstance(x, dict) else x,
                'chat_history': lambda x: x.get('chat_history', '') if isinstance(x, dict) else ''
            } | prompt | model | parser
        )
        
        return chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise