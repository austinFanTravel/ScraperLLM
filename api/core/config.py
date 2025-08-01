from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ScraperLLM API"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Default React dev server
        "https://*.webflow.io",   # Webflow preview
        "https://livewebscraper.com"  # Production domain
    ]
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/query_expander")
    
    # Rate Limiting (requests per minute)
    RATE_LIMIT: int = 60
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security (for future use)
    API_KEY: str = os.getenv("API_KEY", "")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()
