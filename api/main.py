from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routers import status, scrape, query
from .core.config import settings
from .core.logging import setup_logging

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="ScraperLLM API",
    description="API for ScraperLLM - AI-powered web scraping service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status.router, prefix="/api", tags=["Status"])
app.include_router(scrape.router, prefix="/api/scrape", tags=["Scraping"])
app.include_router(query.router, prefix="/api/query", tags=["Query Expansion"])
