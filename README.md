# 🏥 Agentic Healthcare + Hospital Finder

A comprehensive AI-powered healthcare assistant that combines **Retrieval-Augmented Generation (RAG)** for answering clinical questions with a **Search Agent** for finding hospitals and emergency services in real-time.

Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), [OpenAI](https://openai.com/), and [Tavily](https://tavily.com/). Tracing and observability provided by [LangSmith](https://smith.langchain.com/).

## ✨ Features

*   **🩺 Clinical Q&A (RAG)**: Answers questions about diseases (Diabetes, Hypertension, Asthma, etc.) using a curated local knowledge base.
*   **🏥 Hospital Finder**: Locates nearby hospitals, clinics, and emergency centers using real-time web search (Tavily).
*   **🧠 Intelligent Routing**: A master agent automatically decides whether to use the RAG knowledge base or the Hospital Finder tool based on your query.
*   **📂 Custom Knowledge Base**: Upload your own medical text files (`.txt`) directly via the sidebar to expand the agent's knowledge.
*   **📊 Observability**: Integrated with **LangSmith** for real-time tracing and debugging of agent actions.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Orchestration**: LangChain
*   **LLM**: OpenAI GPT-4o-mini
*   **Vector Querying**: FAISS (Facebook AI Similarity Search)
*   **Web Search**: Tavily Search API
*   **Observability**: LangSmith

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Healthcare_rag_agent
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:

```ini
# OpenAI API Key (Required for LLM and Embeddings)
OPENAI_API_KEY="sk-..."

# Tavily API Key (Required for Hospital Search)
TAVILY_API_KEY="tvly-..."

# LangSmith Configuration (Recommended for Tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="Healthcare-RAG"
LANGCHAIN_API_KEY="ls-..."
```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Example Queries
*   *"What are the symptoms of hypertension?"* (Uses RAG)
*   *"Find the best cardiology hospital in New York."* (Uses Hospital Finder)
*   *"How is asthma treated?"* (Uses RAG)
*   *"Where is the nearest emergency room?"* (Uses Hospital Finder)

## 📂 Project Structure

*   **`app.py`**: Main application entry point. Sets up the Streamlit UI, initializes the RAG system, and orchestrates the Master Agent.
*   **`hospital_agent.py`**: Defines the Tavily-powered Hospital Finder tool.
*   **`requirements.txt`**: List of Python dependencies.
*   **`.env`**: Stores sensitive API keys (not included in version control).

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## 📄 License
MIT License
