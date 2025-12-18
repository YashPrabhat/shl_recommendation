# SHL Intelligent Assessment Recommendation System

## 1. Solution Overview
This project implements an intelligent RAG (Retrieval-Augmented Generation) system to recommend SHL assessments based on natural language job descriptions. It moves beyond simple keyword matching by using semantic vector search combined with LLM-based reasoning to ensure a balanced mix of technical (Hard Skills) and behavioral (Soft Skills) assessments.

## 2. Technical Architecture
The solution consists of four modular components:

### A. Data Ingestion (`ingest_data.py`)
- **Tool:** Playwright & BeautifulSoup.
- **Strategy:** Handles dynamic JavaScript loading on the SHL catalog. It filters out "Pre-packaged solutions" and scrapes detailed metadata (Duration, Adaptive Support) by visiting individual product pages.
- **Output:** A clean JSON dataset of 370+ individual assessments.

### B. Vector Database (`vector_store.py`)
- **Model:** `all-MiniLM-L6-v2` (HuggingFace).
- **Storage:** ChromaDB (Local persistent vector store).
- **Context Engineering:** Embeddings are generated from a rich context string combining `Assessment Name`, `Category`, and `Description`. This allows retrieval based on semantic meaning (e.g., "Teamwork" maps to "Personality Tests").
- **Optimization:** Forced CPU execution to ensure compatibility across environments.

### C. RAG & Inference Engine (`rag_engine.py`)
- **Pipeline:** 
  1. **Retrieve:** Fetch top 15 candidates from ChromaDB using semantic similarity.
  2. **Reason:** Pass candidates to Google Gemini (LLM) with a strict prompt enforcing the "Balance" constraint (mix of Knowledge & Skills + Personality).
  3. **Fallback:** Implemented a robustness layer that defaults to raw semantic search if the LLM API experiences downtime or rate limits.

### D. API & Frontend (`main.py` / `app.py`)
- **API:** FastAPI server adhering strictly to the Appendix 2 specifications.
- **Frontend:** Streamlit interface for interactive testing and visualization.

## 3. Optimization & Trade-offs
- **Handling Rate Limits:** The system includes a fallback mechanism. If the Gemini API returns a 429/503 error, the system automatically downgrades to a pure vector search to ensure the API never fails to return a result.
- **Reproducibility:** A virtual environment and standard `requirements.txt` ensure the code runs on any Linux/Mac/Windows machine without GPU dependencies.