"""Search functionality for ScraperLLM.

This module provides interfaces and implementations for searching and retrieving
information from various sources, with support for different search intents
and result formats.
"""

from .base import SearchEngine, SearchResult, SearchIntent, EntityType, Entity
from .serpapi_search import SerpAPISearcher, WebSearcher

__all__ = [
    # Base classes
    "SearchEngine",
    "SearchResult",
    "SearchIntent",
    "EntityType",
    "Entity",
    # Implementations
    "SerpAPISearcher",
    "WebSearcher",  # Alias for backward compatibility
]
