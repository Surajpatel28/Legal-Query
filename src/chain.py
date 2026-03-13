from functools import lru_cache
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_MODEL_NAME, GROQ_API_KEY
from src.retrieval import extract_section, format_docs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    try:
        model = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.2
        )
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise


@lru_cache(maxsize=1)
def get_prompt():
    prompt = PromptTemplate(
        template="""You are legal advisor assistant for question-answering tasks.
        {chat_history}
        Use the following piece of retrieved context to answer the following query.
        {query}

        **Don't answer without referring the context.**
        
        Find Bharatiya Nyaya Sanhita (BNS) Section which is criminal code of India from the context. 
        Focus on entities present in the information. Only add relevant information.
        
        USE BEAUTIFUL MARKDOWN FORMAT and ANSWER like chatbot. 
        
        Provide information in below format:
        A. BNS sections:
                List all sections in below format:
                Section (section number or name): Description \n
        
        B. Punishments:
                Define Punishments in detail for associate section name
        
        C. Legal Advice:
            Define legal advice including police and medical if emergency. Also prvoide
            help line number for India.
        
        Context:
        {context}
        
        """,
        input_variables=["context", "query", "chat_history"]
    )
    return prompt

def create_rag_chain(retrieve_fn, cache_get_fn=None, cache_store_fn=None):
    model = get_model()
    prompt = get_prompt()
    parser = StrOutputParser()
    llm_chain = prompt | model | parser
    
    def run_chain(inputs: dict) -> str:
        
        query = inputs.get("query")
        chat_history = inputs.get("chat_history", "")

        cached_response = None  
        if cache_get_fn is not None:
            cached_response = cache_get_fn(query)

        if cached_response:
            logger.info("CACHE HIT for query: %s", query)
            return cached_response

        logger.info("CACHE MISS → running RAG for query: %s", query)
        docs = retrieve_fn(query)
        section = extract_section(query)

        if section and not docs:
                return (
                    f"Section {section} does not exist in the Bharatiya Nyaya Sanhita (BNS). "
                    "BNS replaced the Indian Penal Code (IPC) and renumbered many sections. "
                    "Please try asking by topic instead, e.g. 'What is the BNS law on theft?'"
                )
        context = format_docs(docs)
        response =  llm_chain.invoke({
            "context": context,
            "query": query, 
            "chat_history": chat_history
            })
        
        if cache_store_fn is not None and response:
            cache_store_fn(query, response)

        return response

    return run_chain