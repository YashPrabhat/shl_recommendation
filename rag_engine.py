import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load API Key from .env
load_dotenv()

# --- CONFIGURATION ---
DB_DIR = "chroma_db"
# Switch to Flash model (Faster & more reliable for free tier)
LLM_MODEL = "gemini-2.5-flash-lite"

class RecommendationEngine:
    def __init__(self):
        # 1. Initialize Retrieval (CPU-forced)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # 2. Initialize LLM (Gemini)
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in .env file")
            
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.2,
            max_retries=2, # Auto-retry if 503 error occurs
            convert_system_message_to_human=True
        )

    def search_and_recommend(self, user_query):
        """
        Full Pipeline: Query -> Retrieve 15 -> LLM Selects Best 5-10 -> Format JSON
        """
        print(f"Processing Query: {user_query}")
        
        # Step A: Retrieval
        docs = self.db.similarity_search(user_query, k=15)
        
        # Create mapping
        candidates = []
        doc_map = {}
        
        for idx, doc in enumerate(docs):
            doc_map[idx] = doc.metadata
            candidates.append(
                f"ID {idx}: Name: {doc.metadata['name']} | Type: {doc.metadata['test_type']} | Desc: {doc.metadata.get('description', '')[:200]}..."
            )
        
        candidates_text = "\n".join(candidates)

        # Step B: Context Engineering
        prompt = PromptTemplate(
            template="""
            You are an expert HR Recruitment Consultant.
            
            USER QUERY: "{query}"
            
            Here are 15 available assessments from the SHL catalog:
            {candidates}
            
            TASK:
            Select a set of 5 to 10 assessments that best match the user's needs.
            
            CRITICAL RULES:
            1. **Relevance**: Must match the job role.
            2. **Balance**: If the query implies a role (like "Manager" or "Developer"), you MUST select a mix of:
               - "Knowledge & Skills" (Hard skills)
               - "Personality & Behavior" (Soft skills/Culture fit)
               - "Ability & Aptitude" (Cognitive fit)
            3. **Quantity**: Minimum 5, Maximum 10.
            
            OUTPUT FORMAT:
            Return ONLY a valid JSON object with a key "selected_ids" containing the list of integer IDs.
            Example: {{ "selected_ids": [0, 2, 5, 8] }}
            """,
            input_variables=["query", "candidates"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        
        final_recommendations = []
        
        try:
            # Try LLM Generation
            response = chain.invoke({"query": user_query, "candidates": candidates_text})
            selected_ids = response.get("selected_ids", [])
            
            # Hydrate selected IDs
            for pid in selected_ids:
                if pid in doc_map:
                    self._add_to_list(final_recommendations, doc_map[pid])
            
            print("AI Selection Successful.")

        except Exception as e:
            print(f"LLM Error/Timeout ({e}). Switching to Fallback Mode.")
            # FALLBACK: If LLM fails, return the top 5 raw semantic matches
            # This ensures the API NEVER returns empty
            for i in range(5):
                if i in doc_map:
                    self._add_to_list(final_recommendations, doc_map[i])

        # Ensure Min 5 / Max 10 constraint
        # If AI picked too few, fill with top search results
        if len(final_recommendations) < 5:
            print("Not enough recommendations. Filling with search results.")
            for i in range(15):
                if len(final_recommendations) >= 5: break
                # Check if this doc is already added (by name)
                current_names = [x['name'] for x in final_recommendations]
                if doc_map[i]['name'] not in current_names:
                    self._add_to_list(final_recommendations, doc_map[i])

        # Cap at 10
        return {"recommended_assessments": final_recommendations[:10]}

    def _add_to_list(self, list_obj, meta):
        """Helper to format and add a document"""
        list_obj.append({
            "url": meta['url'],
            "name": meta['name'],
            "adaptive_support": meta['adaptive_support'],
            "description": meta.get('description', 'No description')[:300],
            "duration": int(meta['duration']),
            "remote_support": meta['remote_support'],
            "test_type": [meta['test_type']]
        })

# --- TEST BLOCK ---
if __name__ == "__main__":
    engine = RecommendationEngine()
    test_query = "Need a Java developer who is good in collaborating with external teams and stakeholders."
    result = engine.search_and_recommend(test_query)
    print(json.dumps(result, indent=2))