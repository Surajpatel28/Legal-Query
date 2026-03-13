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
    template="""
    You are a legal information assistant specialized in the Bharatiya Nyaya Sanhita (BNS).

    Your task is to answer a user's legal query **strictly using the retrieved context provided below**.

    -----------------------------
    RULES
    -----------------------------
    1. Only use information present in the retrieved context.
    2. Do NOT use outside knowledge or assumptions.
    3. If the retrieved context does not contain the relevant BNS section, respond with:
    "The retrieved context does not contain information relevant to this query."
    4. Only include BNS sections that are clearly related to the user's query.
    5. Ignore any retrieved sections that are unrelated to the query.
    6. Do NOT invent section numbers, punishments, or legal interpretations.

    -----------------------------
    USER QUERY
    -----------------------------
    {query}

    -----------------------------
    RETRIEVED CONTEXT
    -----------------------------
    {context}

    -----------------------------
    CHAT HISTORY
    -----------------------------
    {chat_history}

    -----------------------------
    OUTPUT FORMAT (STRICT)
    -----------------------------

    A. Relevant BNS Sections

    List only sections clearly related to the query.

    **Section <number> — <Title if available>**  
    Short explanation derived from the retrieved context.

    Repeat for each relevant section.

    ---

    B. Punishments

    For each section listed above:

    **Section <number> Punishment**

    Explain the punishment exactly as described in the retrieved context.

    If punishment details are missing in the context, write:

    "Punishment details are not available in the retrieved context."

    ---

    C. Legal Guidance

    Provide practical steps someone should follow in this situation.

    Include Indian emergency contacts:

    Police: 100  
    Ambulance: 102  
    National Emergency Helpline: 112

    Guidance should remain general and informational, not legal representation.

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