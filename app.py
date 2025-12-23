
import os
import streamlit as st
from typing import List
from dotenv import load_dotenv

load_dotenv()

# =========================
# SAFE IMPORTS ONLY
# =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

from hospital_agent import hospital_search


# =========================
# API KEY
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing")
    st.stop()


# =========================
# UI
# =========================
st.set_page_config(page_title="Healthcare RAG + Hospital Finder", layout="wide")
st.title("🏥 Healthcare AI Assistant")

st.sidebar.header("📄 Knowledge Base")
uploaded = st.sidebar.file_uploader(
    "Upload .txt files",
    type=["txt"],
    accept_multiple_files=True
)


# =========================
# DOCUMENTS
# =========================
docs = [
    "Diabetes affects blood sugar levels. Symptoms include thirst and fatigue.",
    "Hypertension increases heart disease risk. Normal BP is 120/80 mmHg.",
    "Asthma is a chronic airway inflammation triggered by pollen and dust."
]

if uploaded:
    for f in uploaded:
        try:
            docs.append(f.getvalue().decode("utf-8"))
        except Exception:
            pass


# =========================
# VECTOR STORE
# =========================
with st.spinner("Building knowledge base..."):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks: List[str] = []
    for d in docs:
        chunks.extend(splitter.split_text(d))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )


# =========================
# MANUAL RAG
# =========================
def healthcare_answer(query: str) -> str:
    try:
        # NEW API
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "I don't know based on the available knowledge base."

        context = "\n\n".join(d.page_content for d in retrieved_docs)

        prompt = f"""
Answer the question using ONLY the context below.
If you don't know, say you don't know.

Context:
{context}

Question:
{query}
"""

        response = rag_llm.invoke([HumanMessage(content=prompt)])
        return response.content

    except Exception as e:
        return f"RAG error: {e}"



# =========================
# CHAT UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

user_input = st.chat_input("Ask about health or hospitals (e.g. asthma symptoms, best hospital in Pune)")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    hospital_keywords = [
        "hospital", "clinic", "icu", "ambulance",
        "near me", "nearest", "best hospital"
    ]

    with st.spinner("Thinking..."):
        if any(k in user_input.lower() for k in hospital_keywords):
            answer = hospital_search(user_input)
        else:
            answer = healthcare_answer(user_input)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.history.append(("assistant", answer))
