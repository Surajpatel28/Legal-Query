import logging
import streamlit as st
from src import initialize_rag_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(page_title="LegalQuery", page_icon="⚖️", layout="centered")

st.title("LegalQuery")
st.caption("AI Legal Information Assistant")

st.info("⚠️ Disclaimer: This is for educational purposes only. Not legal advice. Consult a qualified lawyer for your specific situation.")

if "chain" not in st.session_state:
    with st.spinner("Initializing system..."):
        try:
            st.session_state.chain = initialize_rag_system()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            chat_history = "\n\n".join(
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in st.session_state.messages[-6:-1]
            )
            
            with st.spinner("Searching..."):
                response = st.session_state.chain({
                    "query" : prompt,
                    "chat_history" : chat_history
                })
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

with st.sidebar:
    st.header("About")
    st.write("LegalQuery provides information about the Bharatiya Nyaya Sanhita (BNS).")
    st.warning("This is for educational purposes only. Not legal advice.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
