"""
ScraperLLM Web Application
"""
import io
import csv
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from scraper_llm.search_assistant import SearchAssistant
from scraper_llm.core.logging import configure_logging

# Configure logging
configure_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ScraperLLM Web", version="0.1.0")

# Setup templates and static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Request model
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

    @classmethod
    def from_form_data(cls, form_data: dict):
        return cls(
            query=form_data.get('query', ''),
            max_results=int(form_data.get('max_results', 5))
        )

# Initialize search assistant
search_assistant = SearchAssistant(
    model_name="all-mpnet-base-v2",  # or your preferred model
    data_dir="./data/search_assistant",
    use_gpu=False  # Set to True if you have a GPU
)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main search page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search(request: Request):
    """Handle search requests and return HTML response"""
    try:
        # Try to parse JSON data first
        try:
            json_data = await request.json()
            query = json_data.get('query', '')
            num_results = int(json_data.get('max_results', 5))
        except:
            # Fall back to form data if JSON parsing fails
            form_data = await request.form()
            query = form_data.get('query', '')
            num_results = int(form_data.get('max_results', 5))
        
        if not query:
            return HTMLResponse("<div class='error'>Please enter a search query</div>")
        
        # Perform the search asynchronously
        results = await search_assistant.search(
            query=query,
            num_results=num_results
        )
        
        # Format results as HTML
        if not results:
            return HTMLResponse("<div class='no-results'>No results found. Try a different search term.</div>")
            
        results_html = ""
        for result in results:
            # Handle different result formats
            if isinstance(result, dict):
                title = result.get('title') or result.get('metadata', {}).get('title', 'No title')
                url = result.get('url') or result.get('link') or result.get('metadata', {}).get('url', '#')
                snippet = result.get('snippet') or result.get('text') or result.get('description', 'No description available.')
                score = result.get('score', result.get('relevance_score', 0))
            else:
                # Handle object with attributes
                title = getattr(result, 'title', 'No title')
                url = getattr(result, 'url', getattr(result, 'link', '#'))
                snippet = getattr(result, 'snippet', getattr(result, 'text', 'No description available.'))
                score = getattr(result, 'score', getattr(result, 'relevance_score', 0))
            
            results_html += f"""
            <div class='result-item'>
                <h3 class='result-title'>
                    <a href='{url}' target='_blank'>{title}</a>
                </h3>
                <p class='result-url'>{url}</p>
                <p class='result-snippet'>{snippet}</p>
                <div class='result-meta'>
                    <span class='result-score'>Relevance: {float(score):.2f}</span>
                </div>
            </div>
            """
        
        return HTMLResponse(results_html)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        error_msg = f"<div class='error'>Error performing search: {str(e)}</div>"
        return HTMLResponse(error_msg, status_code=500)

@app.post("/api/export-csv")
async def export_csv(request: Request):
    """Export search results as CSV"""
    try:
        # Try to parse JSON data first
        try:
            json_data = await request.json()
            query = json_data.get('query', '')
            num_results = int(json_data.get('max_results', 5))
        except:
            # Fall back to form data if JSON parsing fails
            form_data = await request.form()
            query = form_data.get('query', '')
            num_results = int(form_data.get('max_results', 5))
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query parameter is required"}
            )
        
        # Perform the search asynchronously
        results = await search_assistant.search(
            query=query,
            num_results=num_results
        )
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={"error": "No results found to export"}
            )
        
        # Create a CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Title', 'URL', 'Snippet', 'Relevance Score'])
        
        # Write data
        for result in results:
            # Handle different result formats
            if isinstance(result, dict):
                title = result.get('title') or result.get('metadata', {}).get('title', 'No title')
                url = result.get('url') or result.get('link') or result.get('metadata', {}).get('url', '#')
                snippet = result.get('snippet') or result.get('text') or result.get('description', 'No description available.')
                score = result.get('score', result.get('relevance_score', 0))
            else:
                # Handle object with attributes
                title = getattr(result, 'title', 'No title')
                url = getattr(result, 'url', getattr(result, 'link', '#'))
                snippet = getattr(result, 'snippet', getattr(result, 'text', 'No description available.'))
                score = getattr(result, 'score', getattr(result, 'relevance_score', 0))
            
            writer.writerow([
                title,
                url,
                snippet.replace('\n', ' ').strip(),
                f"{float(score):.4f}"
            ])
        
        # Create response
        response = Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=search_results_{query[:50]}.csv"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to export results: {str(e)}"}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    uvicorn.run(
        "scraper_llm.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
