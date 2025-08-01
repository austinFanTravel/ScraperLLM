from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
import os

# Import your model loading and prediction functions
from ..services.semantic_expander import SemanticExpander
from ..core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the model at startup
model = None
try:
    model = SemanticExpander(settings.MODEL_PATH)
    logger.info("Semantic expansion model loaded successfully")
except Exception as e:
    logger.error(f"Error loading semantic expansion model: {str(e)}")
    # Don't raise here to allow the API to start without the model
    # The endpoints will return 503 if the model isn't available

class QueryExpansionRequest(BaseModel):
    """Request model for query expansion."""
    query: str
    max_expansions: int = 5
    min_confidence: float = 0.7

class QueryExpansionResponse(BaseModel):
    """Response model for query expansion."""
    original_query: str
    expanded_queries: List[Dict[str, Any]]
    
class ModelStatusResponse(BaseModel):
    """Response model for model status check."""
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

def check_model_available():
    """Check if the model is available."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic expansion model is not available. Please check the logs for more information."
        )

@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """
    Check the status of the semantic expansion model.
    Returns whether the model is loaded and basic information about it.
    """
    if model is None:
        return ModelStatusResponse(
            status="error",
            model_loaded=False,
            model_info={"error": "Model failed to load on startup"}
        )
    
    try:
        model_info = model.get_model_info()
        return ModelStatusResponse(
            status="ready",
            model_loaded=True,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return ModelStatusResponse(
            status="error",
            model_loaded=False,
            model_info={"error": str(e)}
        )

@router.post("/expand", response_model=QueryExpansionResponse)
async def expand_query(
    request: QueryExpansionRequest,
    api_key: str = Query(..., description="API key for authentication")
) -> QueryExpansionResponse:
    """
    Expand a search query using the trained semantic expansion model.
    
    - **query**: The original search query to expand
    - **max_expansions**: Maximum number of expanded queries to return (default: 5)
    - **min_confidence**: Minimum confidence score for expanded queries (0.0 to 1.0, default: 0.7)
    - **api_key**: Your API key for authentication
    
    Returns a list of expanded queries with their confidence scores.
    """
    # Check if model is available
    check_model_available()
    
    # TODO: Implement API key validation
    # if not validate_api_key(api_key):
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call the model to expand the query
        expanded = model.expand_query(
            query=request.query,
            max_expansions=request.max_expansions,
            min_confidence=request.min_confidence
        )
        
        return QueryExpansionResponse(
            original_query=request.query,
            expanded_queries=expanded
        )
        
    except Exception as e:
        logger.error(f"Error expanding query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error expanding query: {str(e)}"
        )
