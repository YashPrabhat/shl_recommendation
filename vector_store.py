import json
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
DATA_FILE = "shl_assessments.json"
DB_DIR = "chroma_db"

def create_vector_db():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading {len(data)} assessments...")

    documents = []
    for item in data:
        page_content = f"""
        Assessment Name: {item['name']}
        Category: {', '.join(item['test_type'])}
        Description: {item['description']}
        """
        
        metadata = {
            "name": item['name'],
            "url": item['url'],
            "duration": item['duration'],
            "adaptive_support": item['adaptive_support'],
            "remote_support": item['remote_support'],
            "test_type": ", ".join(item['test_type'])
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print("Initializing AI Model (all-MiniLM-L6-v2)...")
    # FORCE CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    print("Creating Vector Database...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Success! Vector Database created at '{DB_DIR}' with {len(documents)} items.")

def test_retrieval():
    """Simple test to prove the brain works"""
    print("\n--- TEST SEARCH ---")
    
    # FORCE CPU HERE TOO
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    query = "I need a Python developer who can work with teams"
    print(f"Query: {query}")
    
    # Fetch 5 results to be sure
    results = db.similarity_search(query, k=5)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Name: {res.metadata['name']}")
        print(f"Type: {res.metadata['test_type']}")
        print(f"Score: Match found") # Chroma doesn't return score in simple search

if __name__ == "__main__":
    # If the DB folder already exists and has files, skip creation to save time
    if not os.path.exists(DB_DIR):
        create_vector_db()
    else:
        print("Database already exists. Skipping creation.")
        
    test_retrieval()