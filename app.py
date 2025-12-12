# import os
# from dotenv import load_dotenv
# load_dotenv()
# import streamlit as st
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.tools import Tool
# from langchain.agents import initialize_agent, AgentType


# # -------------------------------------------------------
# # Load OpenAI Key from Environment
# # -------------------------------------------------------
# openai_key = os.getenv("OPENAI_API_KEY")

# if not openai_key:
#     st.error("⚠ OPENAI_API_KEY is missing in environment variables!")
#     st.stop()


# # -------------------------------------------------------
# # Streamlit UI Setup
# # -------------------------------------------------------
# st.set_page_config(page_title="Healthcare RAG Agent", layout="wide")
# st.title("🏥 Healthcare Q&A Agent  Asthma diabetes hypertension")
# st.write("Ask any health-related question. Uses RAG + AI agent.")


# # -------------------------------------------------------
# # Sample Healthcare Documents
# # -------------------------------------------------------
# documents = [
#     {
#         "title": "Diabetes Info",
#         "content": """
# Diabetes is a chronic condition affecting blood sugar.
# Symptoms include frequent urination, thirst, and fatigue.
# Treatment includes lifestyle changes, insulin, and diet control.
# """
#     },
#     {
#         "title": "Blood Pressure Guide",
#         "content": """
# High blood pressure (hypertension) increases risk of heart disease.
# Normal BP is around 120/80 mmHg.
# Reduce sodium intake, exercise regularly, and manage stress.
# """
#     },
#     {
#         "title": "Heart Attack Symptoms",
#         "content": """
# Heart attack symptoms include chest pain, shortness of breath,
# pain spreading to shoulder or arm, nausea, and sweating.
# """
#     }
# ]

# all_texts = [d["content"] for d in documents]


# # -------------------------------------------------------
# # Split documents
# # -------------------------------------------------------
# splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# chunks = []
# for txt in all_texts:
#     chunks.extend(splitter.split_text(txt))


# # -------------------------------------------------------
# # Vector DB
# # -------------------------------------------------------
# embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
# vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
# retriever = vectorstore.as_retriever()


# # -------------------------------------------------------
# # RAG RetrievalQA
# # -------------------------------------------------------
# llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     return_source_documents=True
# )


# # -------------------------------------------------------
# # RAG Function
# # -------------------------------------------------------
# def healthcare_search(query: str):
#     """RAG search function"""
#     response = qa_chain.invoke({"query": query})
#     return response["result"]


# # -------------------------------------------------------
# # Tool for Agent
# # -------------------------------------------------------
# healthcare_tool = Tool(
#     name="HealthcareSearchEngine",
#     func=healthcare_search,
#     description="Useful for answering healthcare questions using RAG."
# )


# # -------------------------------------------------------
# # Agent
# # -------------------------------------------------------
# agent = initialize_agent(
#     tools=[healthcare_tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False
# )


# # -------------------------------------------------------
# # Streamlit Chat UI
# # -------------------------------------------------------
# if "chat_history" in st.session_state:
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])
# else:
#     st.session_state.chat_history = []


# query = st.chat_input("Ask a healthcare question...")

# if query:
#     # Display user message
#     st.session_state.chat_history.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.write(query)

#     # Agent Response
#     with st.chat_message("assistant"):
#         answer = agent.run(query)
#         st.write(answer)

#     st.session_state.chat_history.append({"role": "assistant", "content": answer})



###############################################
# app.py
import os
from typing import List
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # optional, reads .env if present

# LangChain imports for RAG + agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Import hospital tool from hospital_agent.py
from hospital_agent import hospital_tool, hospital_search  # hospital_tool is a LangChain Tool

# -------------------------
# Load keys
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing from environment variables. Set it and restart.")
    st.stop()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # used by hospital_tool internally (hospital_agent.py reads env)

# -------------------------
# Streamlit UI layout
# -------------------------
st.set_page_config(page_title="Agentic Healthcare + Hospital Finder", layout="wide")
st.title("🏥 Agentic AI — Healthcare RAG + Hospital Finder")
st.write(
    "This app has two separate agents under one master interface:\n\n"
    "- Healthcare RAG Agent (answers clinical questions from local docs)\n"
    "- Hospital Finder Agent (finds hospitals via Tavily Search)\n\n"
    "Ask a question and the master agent will pick the right tool."
)

# Optional: allow user to upload additional text docs to augment RAG KB
st.sidebar.header("Knowledge Base")
uploaded = st.sidebar.file_uploader("Upload .txt files to add to KB (optional)", accept_multiple_files=True, type=["txt"])
sample_docs = [
    {
        "title": "Diabetes Info",
        "content": "Diabetes is a chronic condition affecting blood sugar. Symptoms include frequent urination, thirst, and fatigue. Treatment includes lifestyle changes, insulin, and diet control."
    },
    {
        "title": "Hypertension Guide",
        "content": "Hypertension increases risk of heart disease. Normal BP is 120/80 mmHg. Reduce sodium intake, exercise, weight control and medication when needed."
    },
    {
        "title": "Asthma Basics",
        "content": "Asthma is chronic airway inflammation. Triggers: pollen, dust, cold air. Treatment: inhaled steroids and bronchodilators."
    },
]

# read uploaded files (if any)
if uploaded:
    for f in uploaded:
        try:
            txt = f.getvalue().decode("utf-8")
        except Exception:
            txt = ""
        sample_docs.append({"title": f.name, "content": txt})

# -------------------------
# Build RAG (embeddings + FAISS)
# -------------------------
with st.spinner("Preparing knowledge base (embeddings). This may take a few seconds..."):
    texts: List[str] = [d["content"] for d in sample_docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm_for_qa = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0, model_name="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_qa,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

# Healthcare tool for RAG (wrap .invoke)
def healthcare_search(query: str) -> str:
    try:
        resp = qa_chain.invoke({"query": query})
        answer = resp.get("result") or resp.get("output_text") or str(resp)
        # attach short source snippets (optional)
        srcs = resp.get("source_documents") or []
        if srcs:
            snippets = []
            for i, sd in enumerate(srcs[:3], start=1):
                try:
                    snippet_text = sd.page_content[:180] + "..." if hasattr(sd, "page_content") else str(sd)[:180] + "..."
                except Exception:
                    snippet_text = str(sd)[:180] + "..."
                snippets.append(f"Source {i}: {snippet_text}")
            answer = answer + "\n\n" + "\n".join(snippets)
        return answer
    except Exception as e:
        return f"RAG error: {e}"

healthcare_tool = Tool(
    name="HealthcareRAG",
    func=healthcare_search,
    description="Answer clinical and healthcare questions using the local knowledge base."
)

# -------------------------
# Master agent (combines both tools)
# -------------------------
# Create a fresh LLM for the master agent
master_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0, model_name="gpt-4o-mini")

# we import hospital_tool from hospital_agent.py; create tools list
tools_for_master = [healthcare_tool, hospital_tool]

master_agent = initialize_agent(
    tools=tools_for_master,
    llm=master_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# -------------------------
# Chat UI
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# show previous messages
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

prompt = st.chat_input("Ask a healthcare or hospital question (e.g., 'symptoms of asthma' or 'best cardiology hospital in Pune')")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # simple heuristic: if question is explicitly hospital-related, call hospital_tool directly (faster),
    # otherwise let the master agent decide. This is optional — remove heuristic to always let agent decide.
    lowered = prompt.lower()
    hospital_keywords = ["hospital", "clinic", "emergency", "near me", "nearest", "nearest hospital", "hospitals in", "best hospital", "icu", "ambulance"]
    use_hospital_direct = any(k in lowered for k in hospital_keywords)

    with st.spinner("Agent is thinking..."):
        if use_hospital_direct:
            # call hospital tool directly to avoid extra agent reasoning step
            try:
                answer_text = hospital_search(prompt)
            except Exception as e:
                answer_text = f"Hospital tool error: {e}"
        else:
            # Let the agent orchestrate
            try:
                answer_text = master_agent.run(prompt)
            except Exception as e:
                # fallback to RAG
                fallback = healthcare_search(prompt)
                answer_text = f"Agent error: {e}\n\nFalling back to RAG answer:\n\n{fallback}"

    with st.chat_message("assistant"):
        st.write(answer_text)
    st.session_state.history.append(("assistant", answer_text))
