from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.config import GROQ_MODEL_NAME, GROQ_API_KEY
from src.retrieval import format_docs
from langchain.agents import create_agent
try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = RecursionError

def get_model():
    try:
        model = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.2
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
- If the question is not related to Indian law, BNS, crimes, or legal topics, respond only with: "I can only answer questions related to Indian law and the Bharatiya Nyaya Sanhita (BNS)."
- BNS sections are the primary authority for Indian criminal law. Always prefer them.
- If Wikipedia context is provided, treat it as background or definitional support only.
- If web search context is provided, treat it as supplementary context only.
- Clearly distinguish in your answer which source category the information comes from.
- If no reliable evidence was found, say so explicitly.
- Explain in simple, clear language that anyone can understand.
- Use conversation history to understand follow-up questions.

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

def build_research_agent(tools):
    model = get_model()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a legal research assistant specialising in Indian law and the Bharatiya Nyaya Sanhita (BNS). "
            "STRICT RULES — follow exactly:\n"
            "1. If the question is NOT about Indian law, BNS, crimes, or legal topics: respond with exactly OFF_TOPIC and call NO tools.\n"
            "2. Call bns_search EXACTLY ONCE per question. NEVER call bns_search a second time regardless of the result.\n"
            "3. After bns_search returns any result, STOP IMMEDIATELY — do not call any other tool unless the result is completely empty.\n"
            "4. If bns_search result is completely empty (no text), you may call wikipedia ONCE for background context.\n"
            "5. Only call tavily_search if BOTH bns_search AND wikipedia returned nothing useful.\n"
            "6. Never call the same tool twice. Each tool may be used at most once per question.\n"
            "7. If the user asks for a specific section number that does not appear in the bns_search result, do NOT retry — use whatever was returned."
        ),
        debug=True
    )
    return agent

def create_react_rag_chain(tools):
    research_agent = build_research_agent(tools)
    model = get_model()
    prompt = get_prompt()
    parser = StrOutputParser()

    def run_chain(inputs: dict) -> str:
        
        query = inputs.get("query")
        chat_history = inputs.get("chat_history", "")

        try:
            agent_result = research_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 5}
            )
        except GraphRecursionError:
            return "I was unable to find a definitive answer for your query in the BNS knowledge base. The section you asked about may not exist in the Bharatiya Nyaya Sanhita, or may be numbered differently. Please try rephrasing your question or ask about the topic (e.g., 'theft' instead of 'Section 379')."

        # Check if agent flagged query as off-topic (no tools called, last message is OFF_TOPIC)
        last_msg = agent_result['messages'][-1]
        if hasattr(last_msg, 'content') and 'OFF_TOPIC' in last_msg.content:
            return "I'm a legal information assistant focused on the Bharatiya Nyaya Sanhita (BNS) and Indian law. I can only answer questions related to Indian criminal law, BNS sections, crimes, and punishments. Please ask a legal question."

        evidence_parts = []
        for msg in agent_result['messages']:
            if hasattr(msg, 'name') and msg.content:
                evidence_parts.append(msg.content)

        combined_evidence = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "No evidence gathered."

        chain = prompt | model | parser
        return chain.invoke({
            'context' : combined_evidence,
            'query' : query,
            'chat_history' : chat_history
        })
    return run_chain