from functools import lru_cache
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from src.config import GROQ_MODEL_NAME, GROQ_API_KEY
from src.retrieval import extract_section, format_docs
from src.utils import load_prompt_template

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


@lru_cache(maxsize=1)
def get_prompt(path):
    return PromptTemplate(
        template=load_prompt_template(path)
    )


@lru_cache(maxsize=1)
def get_wiki_tool():
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=2000
        )
    )


def create_rag_chain(retrieve_fn, cache_get_fn=None, cache_store_fn=None):

    model = get_model()
    parser = StrOutputParser()

    rag_prompt = get_prompt("./prompt_templates/bns_prompt.txt")
    router_prompt = get_prompt("./prompt_templates/router_prompt.txt")

    rag_chain = rag_prompt | model | parser
    router_chain = router_prompt | model | parser

    wiki_tool = get_wiki_tool()

    def run_chain(inputs: dict):

        query = inputs.get("query")
        chat_history = inputs.get("chat_history", "")

        #  CACHE 
        if cache_get_fn:
            cached = cache_get_fn(query)
            if cached:
                logger.info("CACHE HIT → %s", query)
                return cached

        logger.info("CACHE MISS → %s", query)

        #  ROUTER 
        route = router_chain.invoke({"query": query}).strip()
        logger.info("ROUTER → %s", route)

        response = None

        #  NONE 
        if route == "NONE":
            response = "This assistant only answers questions related to criminal law and Bharatiya Nyaya Sanhita."

        #  WIKI 
        elif route == "WIKI":
            wiki_context = wiki_tool.run(query)

            response = model.invoke(
                f"User query:\n{query}\n\nExplain using this context:\n\n{wiki_context}"
            ).content

        #  BNS 
        elif route == "BNS":

            docs = retrieve_fn(query)
            section = extract_section(query)

            if section and not docs:
                response = (
                    f"Section {section} does not exist in the Bharatiya Nyaya Sanhita (BNS). "
                    "BNS replaced the Indian Penal Code (IPC). "
                    "Try asking by topic instead."
                )

            else:
                context = format_docs(docs)

                response = rag_chain.invoke({
                    "context": context,
                    "query": query,
                    "chat_history": chat_history
                })

        #  CACHE STORE 
        if cache_store_fn and response:
            cache_store_fn(query, response)

        return response

    return run_chain