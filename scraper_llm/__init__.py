"""ScraperLLM: A web scraping and information extraction tool with NER capabilities."""

__version__ = "0.1.0"

from loguru import logger
from .core.logging import configure_logging

# Configure logging when the package is imported
configure_logging()

logger.info(f"ScraperLLM v{__version__} initialized")
