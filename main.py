from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from rag_engine import RecommendationEngine

# Initialize App & Engine
app = FastAPI(title="SHL Recommendation API")
engine = RecommendationEngine()

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Job description or search query")

class AssessmentItem(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# --- ENDPOINTS ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return engine.search_and_recommend(request.query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)