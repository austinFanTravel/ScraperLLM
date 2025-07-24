"""
ScraperLLM Web Application
"""
import io
import csv
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Import your search assistant
from scraper_llm.search_assistant import SearchAssistant
from scraper_llm.core.logging import configure_logging

# --- Configuration ---
class Settings(BaseSettings):
    app_name: str = "ScraperLLM API"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

# Initialize settings
settings = Settings()

# Configure logging
configure_logging(log_level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Setup templates and static files
app.mount(
    "/static", 
    StaticFiles(directory=Path(__file__).parent / "static"), 
    name="static"
)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(5, ge=1, le=50)
    search_type: str = Field("web", pattern="^(web|news|products)$")

class SearchResult(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: str
    score: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_type: str

# Initialize search assistant
search_assistant = SearchAssistant(
    model_name="all-mpnet-base-v2",
    data_dir="./data/search_assistant",
    use_gpu=False
)

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main search page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search", response_model=SearchResponse)
async def search(
    request: Request,
    query: str = Form(...),
    max_results: int = Form(5),
    search_type: str = Form("web")
):
    """Handle search API requests"""
    try:
        logger.info(f"Search request - Query: {query}, Max Results: {max_results}, Type: {search_type}")
        
        # Validate inputs
        search_req = SearchRequest(
            query=query,
            max_results=max_results,
            search_type=search_type
        )
        
        # Perform search (replace with your actual search logic)
        results = await perform_search(
            query=search_req.query,
            max_results=search_req.max_results,
            search_type=search_req.search_type
        )
        
        return {
            "query": search_req.query,
            "results": results,
            "total_results": len(results),
            "search_type": search_req.search_type
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/csv")
async def export_csv(
    request: Request,
    query: str = Form(...),
    max_results: int = Form(5),
    search_type: str = Form("web")
):
    """Export search results as CSV"""
    try:
        # Perform search
        results = await perform_search(query, max_results, search_type)
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Title", "URL", "Snippet"])
        
        # Write data
        for result in results:
            writer.writerow([
                result.get("title", ""),
                result.get("url", ""),
                result.get("snippet", "")
            ])
        
        # Return as file download
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scraperllm_export_{query[:20]}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate CSV export")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": settings.app_version}

# --- Helper Functions ---
async def perform_search(query: str, max_results: int = 5, search_type: str = "web") -> List[Dict[str, Any]]:
    """
    Perform a search using the search assistant
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_type: Type of search (web, news, products)
        
    Returns:
        List of search results
    """
    try:
        # This is a placeholder - replace with your actual search logic
        # For example, using your SearchAssistant class:
        # results = search_assistant.search(query, max_results=max_results, search_type=search_type)
        
        # Mock results for demonstration
        return [
            {
                "title": f"Example Result for '{query}' (1)",
                "url": "https://example.com/1",
                "snippet": f"This is a sample result for the query: {query}",
                "score": 0.95
            },
            {
                "title": f"Example Result for '{query}' (2)",
                "url": "https://example.com/2",
                "snippet": f"Another sample result for: {query}",
                "score": 0.85
            }
        ][:max_results]
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
