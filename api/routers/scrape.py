from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional
from pydantic import BaseModel, HttpUrl
import httpx
from bs4 import BeautifulSoup
import logging

from ..core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class ScrapeRequest(BaseModel):
    """Request model for the scrape endpoint."""
    url: HttpUrl
    selectors: Optional[Dict[str, str]] = None
    extract: Optional[List[str]] = None
    javascript: bool = False

class ScrapeResponse(BaseModel):
    """Response model for the scrape endpoint."""
    url: str
    status_code: int
    content: Optional[Dict] = None
    error: Optional[str] = None

async def fetch_url(url: str) -> str:
    """
    Fetch the content of a URL using httpx.
    
    Args:
        url: The URL to fetch
        
    Returns:
        The HTML content as a string
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching URL: {str(e)}")

def extract_content(html: str, selectors: Optional[Dict[str, str]] = None) -> Dict:
    """
    Extract content from HTML using BeautifulSoup and CSS selectors.
    
    Args:
        html: The HTML content to parse
        selectors: Dictionary of {field_name: css_selector} for extraction
        
    Returns:
        Dictionary containing the extracted content
    """
    soup = BeautifulSoup(html, 'html.parser')
    result = {}
    
    if not selectors:
        # If no selectors provided, return basic page info
        return {
            "title": getattr(soup.title, 'string', ''),
            "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        }
    
    for field, selector in selectors.items():
        try:
            elements = soup.select(selector)
            if len(elements) == 1:
                result[field] = elements[0].get_text().strip()
            else:
                result[field] = [el.get_text().strip() for el in elements]
        except Exception as e:
            logger.warning(f"Error extracting {field} with selector {selector}: {str(e)}")
            result[field] = None
    
    return result

@router.post("/", response_model=ScrapeResponse)
async def scrape_website(
    request: ScrapeRequest,
    api_key: str = Query(..., description="API key for authentication")
) -> ScrapeResponse:
    """
    Scrape a website and extract content using CSS selectors.
    
    - **url**: The URL of the website to scrape
    - **selectors**: Dictionary of {field_name: css_selector} for content extraction
    - **javascript**: Whether to use a headless browser (not implemented yet)
    - **api_key**: Your API key for authentication
    
    Returns the scraped content structured according to the provided selectors.
    """
    # TODO: Implement API key validation
    # if not validate_api_key(api_key):
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        if request.javascript:
            # TODO: Implement headless browser support
            raise HTTPException(
                status_code=501,
                detail="JavaScript rendering is not yet implemented"
            )
        
        # Fetch the URL
        html = await fetch_url(str(request.url))
        
        # Extract content based on selectors
        content = extract_content(html, request.selectors)
        
        return ScrapeResponse(
            url=str(request.url),
            status_code=200,
            content=content
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scrape_website: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )
