import pandas as pd
from rag_engine import RecommendationEngine

# --- CONFIGURATION ---
# If you have the "Labelled Train Set" CSV from the PDF, 
# save it as 'ground_truth.csv' and run this script.
GROUND_TRUTH_FILE = "ground_truth.csv" 

def calculate_recall_at_k(engine, test_data, k=10):
    total_recall = 0
    num_queries = len(test_data)
    
    print(f"Starting Evaluation on {num_queries} queries (K={k})...")
    
    for idx, row in test_data.iterrows():
        query = row['Query']
        # Ground truth URLs (assuming comma-separated in CSV or just one per row)
        # Adjust logic depending on actual CSV format provided in PDF
        actual_urls = [row['Assessment_url'].strip()] 
        
        # Get System Predictions
        try:
            result = engine.search_and_recommend(query)
            recommendations = result.get("recommended_assessments", [])
            predicted_urls = [rec['url'] for rec in recommendations[:k]]
            
            # Calculate Intersection
            # How many relevant items did we find?
            relevant_retrieved = len(set(predicted_urls).intersection(set(actual_urls)))
            
            # Recall Formula: (Relevant_Retrieved) / (Total_Relevant)
            # If total_relevant is 0, recall is 0
            if len(actual_urls) > 0:
                recall = relevant_retrieved / len(actual_urls)
            else:
                recall = 0
                
            total_recall += recall
            print(f"Query {idx+1}: Recall@{k} = {recall:.2f}")
            
        except Exception as e:
            print(f"Error evaluating query {idx+1}: {e}")
    
    # Mean Recall
    mean_recall = total_recall / num_queries
    print(f"\n--- FINAL RESULTS ---")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    return mean_recall

if __name__ == "__main__":
    # Create dummy data if file doesn't exist, just to show the code works
    import os
    if not os.path.exists(GROUND_TRUTH_FILE):
        print("Warning: 'ground_truth.csv' not found.")
        print("Creating a dummy test set to demonstrate evaluation logic...")
        data = {
            "Query": ["Need a Java developer", "Sales manager needed"],
            "Assessment_url": [
                "https://www.shl.com/products/product-catalog/view/java-8-new/",
                "https://www.shl.com/products/product-catalog/view/sales-manager/" # Example URL
            ]
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(GROUND_TRUTH_FILE)

    engine = RecommendationEngine()
    calculate_recall_at_k(engine, df, k=10)