import pandas as pd
import time
from rag_engine import RecommendationEngine

# --- CONFIGURATION ---
OUTPUT_CSV = "submission.csv"

# The 9 Test Queries (Derived from PDF Appendix/Dataset)
TEST_QUERIES = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
    "Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests",
    "I need a sales manager who can drive revenue and manage a large team.",
    "Looking for a graduate trainee with strong numerical and verbal reasoning skills.",
    "We need a customer service representative who remains calm under pressure.",
    "Hiring a senior project manager with agile certification and risk management skills.",
    "Need a mechanical engineer with understanding of thermodynamics and fluid mechanics.",
    "Looking for a marketing specialist with digital marketing and content creation skills."
]

def generate_csv():
    print("Initializing Engine...")
    engine = RecommendationEngine()
    
    submission_rows = []

    print(f"Processing {len(TEST_QUERIES)} queries...")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"[{i+1}/{len(TEST_QUERIES)}] predicting for: {query[:50]}...")
        
        # Get recommendations
        try:
            result = engine.search_and_recommend(query)
            recs = result.get("recommended_assessments", [])
            
            # Format requires: Query Name | Recommendation URL
            # The PDF format example implies repeating the query name for each recommendation
            query_label = f"Query {i+1}" 
            
            for rec in recs:
                submission_rows.append({
                    "Query": query_label, # Or the actual query text if they prefer
                    "Assessment_url": rec['url']
                })
                
            # Sleep to avoid hitting API Rate Limits again
            time.sleep(2) 
            
        except Exception as e:
            print(f"Error on query {i}: {e}")

    # Create DataFrame
    df = pd.DataFrame(submission_rows)
    
    # Verify columns
    print("\nSample Output:")
    print(df.head())
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUCCESS: Submission file saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_csv()