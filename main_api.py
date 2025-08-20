# Root endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uvicorn
import sys
import os

# Add the current directory to Python path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your new orchestrator
from new_agricultural_orchestrator import run_agricultural_advisor

app = FastAPI()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    location: Optional[str] = None
    crop_type: Optional[str] = None

class QueryResponse(BaseModel):
    status: str
    query: str
    timestamp: str
    results: Dict[str, Any]
    report: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    try:
        print("Agricultural orchestrator ready")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        raise

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# Agricultural Orchestrator endpoint
@app.post("/agricultural-query", response_model=QueryResponse)
async def process_agricultural_query(request: QueryRequest):
    """Process agricultural advisory query using multi-agent orchestrator"""
    try:
        # Build the query string with location and crop type if provided
        query = request.query
        if request.location:
            query += f" in {request.location}"
        if request.crop_type:
            query += f" for {request.crop_type}"
        
        # Run the orchestrator function
        result = await run_agricultural_advisor(query)
        
        # Extract report if available
        report = None
        if isinstance(result, dict) and 'final_report' in result:
            report = result['final_report']
        
        return QueryResponse(
            status="success",
            query=request.query,
            timestamp=datetime.now().isoformat(),
            results=result,
            report=report
        )
        
    except Exception as e:
        print(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Welcome to Agri-Yodha API",
        "description": "AI-Powered Agricultural Advisory System",
        "docs": "/docs",
        "health": "/health"
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )