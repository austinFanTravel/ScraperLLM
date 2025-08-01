from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import platform
import psutil

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
async def get_status() -> Dict[str, Any]:
    """
    Get the current status and health of the API.
    Returns basic system information and service status.
    """
    try:
        # Get system information
        system_info = {
            "status": "operational",
            "system": {
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
                "cpu": {
                    "percent": psutil.cpu_percent(),
                    "count": psutil.cpu_count(),
                },
            },
            "service": {
                "name": "ScraperLLM API",
                "status": "operational",
            },
        }
        return system_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system status: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for load balancers and monitoring.
    Returns a simple status message if the service is running.
    """
    return {"status": "healthy"}
