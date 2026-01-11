import streamlit as st
import os
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import ingest

# --- Configuration ---
DB_CONNECTION = "postgresql+psycopg2://postgres:secret@localhost:5432/postgres"
COLLECTION_NAME = "email_vectors"
LLM_MODEL = "llama3"

st.set_page_config(page_title="Local Email RAG", layout="wide")

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('sender', 'Unknown')} - {d.metadata.get('subject', 'No Subject')} ({d.metadata.get('date', 'Unknown')})]\nLink: {d.metadata.get('link', '#')}\n{d.page_content}" 
        for d in docs
    )

# --- Sidebar ---
with st.sidebar:
    st.header("🔐 Secure Login")
    if "user_email" not in st.session_state:
        if st.button("Sign in with Google"):
            with st.spinner("Authenticating..."):
                try:
                    user_email, count = ingest.ingest_emails_for_user(limit=50)
                    st.session_state["user_email"] = user_email
                    st.session_state["messages"] = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
    else:
        st.success(f"User: **{st.session_state['user_email']}**")
        
        # REFRESH BUTTON (Crucial for your "Shivang" email)
        if st.button("🔄 Refresh Inbox"):
            with st.spinner("Fetching latest emails..."):
                user_email, count = ingest.ingest_emails_for_user(limit=50)
                st.toast(f"Inbox updated! Scanned {count} emails.")
                
        if st.button("Logout"):
            del st.session_state["user_email"]
            if os.path.exists("token.json"): os.remove("token.json")
            st.rerun()

    # --- PRECISION FILTERS ---
    st.divider()
    st.header("🎯 Precision Filters")
    st.info("Use these to force the AI to find specific emails.")
    
    sender_filter = st.text_input("Filter by Sender Name", placeholder="e.g. Shivang")
    subject_filter = st.text_input("Filter by Subject", placeholder="e.g. Test")

# --- Main App ---
st.title("📧 Chat with your Inbox")

if "user_email" not in st.session_state:
    st.stop()

current_user = st.session_state["user_email"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"Ask about {current_user}'s emails..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        vector_store = get_vectorstore()
        
        # 1. RETRIEVE BROADLY (Get 60 candidates)
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 60, 
                "filter": {"user_id": current_user} 
            }
        )
        all_docs = retriever.invoke(prompt)
        
        # 2. APPLY METADATA FILTERS (Python Logic)
        filtered_docs = []
        for d in all_docs:
            sender_match = True
            if sender_filter:
                if sender_filter.lower() not in d.metadata.get('sender', '').lower():
                    sender_match = False
            
            subject_match = True
            if subject_filter:
                if subject_filter.lower() not in d.metadata.get('subject', '').lower():
                    subject_match = False
            
            if sender_match and subject_match:
                filtered_docs.append(d)
        
        # 3. LIMIT CONTEXT (Take top 15 survivors)
        final_docs = filtered_docs[:50]
        
        if not final_docs:
            if sender_filter or subject_filter:
                response = f"I searched 60 emails but found none matching: Sender='{sender_filter}'."
            else:
                response = "I couldn't find any relevant emails."
            message_placeholder.markdown(response)
        else:
            # 4. GENERATE ANSWER
            llm = ChatOllama(model=LLM_MODEL)
            template = """Answer based ONLY on the filtered emails below.
            
            Context:
            {context}
            
            Question: {question}
            """
            chat_prompt = ChatPromptTemplate.from_template(template)
            chain = chat_prompt | llm
            
            context_text = format_docs(final_docs)
            
            with st.expander(f"View {len(final_docs)} Source Emails"):
                st.text(context_text)

            full_response = ""
            for chunk in chain.stream({"context": context_text, "question": prompt}):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            response = full_response

    st.session_state["messages"].append({"role": "assistant", "content": response})