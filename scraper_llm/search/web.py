"""Web search functionality for ScraperLLM."""
import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from loguru import logger

from .base import SearchEngine, SearchResult, SearchIntent, EntityType
from ..utils.ner import extract_entities_async
from ..core.config import get_settings
from ..utils.ner import extract_entities
import ssl

# Get application settings
settings = get_settings()

# Default headers for web requests
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Search engine configurations
SEARCH_ENGINES = {
    "google": {
        "url": "https://www.google.com/search",
        "params": {
            "q": "{query}",
            "num": "{max_results}",
            "hl": "en",
            "gl": "us",
            "start": "{start}",
        },
        "result_selector": "div.g",
        "title_selector": "h3",
        "link_selector": "a[href]",
        "snippet_selector": "div.IsZvec",
        "next_page_selector": "a#pnnext",
        "pagination": {
            "param": "start",
            "increment": 10,
        },
        "rate_limit": (0.5, 2.0),  # min/max delay between requests (seconds)
    },
    "bing": {
        "url": "https://www.bing.com/search",
        "params": {
            "q": "{query}",
            "count": "{max_results}",
            "first": "{start}",
            "form": "QBLH",
            "setlang": "en-us",
        },
        "result_selector": "li.b_algo",
        "title_selector": "h2",
        "link_selector": "a[href]",
        "snippet_selector": "div.b_caption p",
        "pagination": {
            "param": "first",
            "increment": 10,
        },
        "rate_limit": (0.8, 2.5),
    },
    "duckduckgo": {
        "url": "https://html.duckduckgo.com/html/",
        "method": "POST",
        "data": {
            "q": "{query}",
            "b": "",
            "kl": "us-en",
            "df": "",
        },
        "headers": {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://html.duckduckgo.com",
            "Referer": "https://html.duckduckgo.com/",
        },
        "result_selector": "div.result",
        "title_selector": "h2 a",
        "link_selector": "h2 a",
        "snippet_selector": "a.result__snippet",
        "pagination": {
            "selector": "div.nav-link",
            "link_selector": "a[rel='next']",
        },
        "rate_limit": (1.0, 3.0),
    },
}

# Known search parameters for different search intents
SEARCH_INTENTS = {
    "person": {
        "google": {"tbm": "nws"},  # News results for people
        "bing": {"qft": "+filterui:age-lt1440"},  # Last 24 hours
    },
    "location": {
        "google": {"tbm": "maps"},  # Maps results for locations
        "bing": {"qft": "+filterui:local_aware"},  # Local results
    },
    "thing": {
        "google": {"tbm": ""},  # Regular web search
        "bing": {"qft": ""},  # Regular web search
    },
}

class WebSearcher(SearchEngine):
    """A web search client that can fetch and parse search results.
    
    This class provides search functionality using various search engines and
    includes rate limiting, retries, and entity extraction capabilities.
    """
    
    def __init__(self, **kwargs):
        """Initialize the web searcher.
        
        Args:
            **kwargs: Additional configuration options
                - user_agent: Custom user agent string
                - max_retries: Maximum number of retry attempts (default: 3)
                - timeout: Request timeout in seconds (default: 30)
                - rate_limit_delay: Delay between requests in seconds (default: 1.0)
                - extract_entities: Whether to extract entities from results (default: True)
        """
        super().__init__(**kwargs)
        self.user_agent = kwargs.get("user_agent", UserAgent().chrome)
        self.max_retries = int(kwargs.get("max_retries", 3))
        self.timeout = float(kwargs.get("timeout", 30.0))
        self.rate_limit_delay = float(kwargs.get("rate_limit_delay", 1.0))
        self.extract_entities = kwargs.get("extract_entities", True)
        self.last_request_time = 0
        self.session = None
        
    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests.
        
        This prevents overwhelming the search engine with too many requests
        in a short period of time.
        """
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            delay = self.rate_limit_delay - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
        self.last_request_time = time.time()
        
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if self.session and not self.session.closed:
            if self.session._connector is not None and self.session._connector_owner:
                self.session._connector._close()
            self.session._connector = None
            
    async def _search_google(
        self, query: str, max_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search using Google.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-100)
            **kwargs: Additional search parameters
                - lang: Language code (e.g., 'en', 'es', 'fr')
                - country: Country code (e.g., 'us', 'uk', 'ca')
                - tbs: Time-based search (e.g., 'qdr:h' for past hour, 'qdr:d' for past day)
                
        Returns:
            List of search results
        """
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
            
        try:
            # Prepare search parameters
            params = SEARCH_ENGINES["google"]["params"].copy()
            params["q"] = query
            params["num"] = max_results
            
            # Create a custom SSL context that verifies certificates
            ssl_context = ssl.create_default_context()
            
            # Make the request with SSL verification enabled
            conn = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(
                connector=conn, 
                timeout=timeout,
                headers={"User-Agent": self.user_agent}
            ) as session:
                # Make the request
                async with session.get(
                    SEARCH_ENGINES["google"]["url"], 
                    params=params,
                    ssl=ssl_context
                ) as response:
                    response.raise_for_status()
                    html = await response.text()
            
            # Parse the HTML response
            soup = BeautifulSoup(html, "html.parser")
            results = soup.select(SEARCH_ENGINES["google"]["result_selector"])
            
            # Extract results
            search_results = []
            for result in results:
                title = result.select_one(SEARCH_ENGINES["google"]["title_selector"]).text.strip()
                link = result.select_one(SEARCH_ENGINES["google"]["link_selector"])["href"]
                snippet = result.select_one(SEARCH_ENGINES["google"]["snippet_selector"]).text.strip()
                search_results.append(SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet
                ))
                
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
            
    async def search_async(
        self, 
        query: str, 
        max_results: int = 10,
        intent: Optional[SearchIntent] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search the web asynchronously.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-100)
            intent: Optional search intent (person, location, thing)
            **kwargs: Additional search parameters
                - timeout: Override the default timeout
                - extract_entities: Whether to extract entities from results
                - search_engine: Override the default search engine
                
        Returns:
            List of search results
            
        Raises:
            ValueError: If max_results is not between 1 and 100
            aiohttp.ClientError: If there's an error making the request
        """
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
            
        # Use the appropriate search method based on intent or default to google
        search_engine = kwargs.get('search_engine', 'google')
        
        # Apply rate limiting
        await self._rate_limit()
        
        try:
            results = await self._search_google(
                query=query,
                max_results=max_results,
                **kwargs
            )
            
            # Extract entities if enabled
            if kwargs.get('extract_entities', self.extract_entities):
                await self._extract_entities_async(results)
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
            
    async def _extract_entities_async(self, results: List[SearchResult]) -> None:
        """Extract named entities from search results asynchronously.
        
        Args:
            results: List of search results to process
            
        Note:
            This modifies the results in place by adding entities to each result.
        """
        if not results:
            return
            
        # Extract text from results
        texts = [f"{r.title}. {r.snippet}" for r in results]
        
        try:
            # Extract entities in batches
            entities_list = await extract_entities_async(texts)
            
            # Add entities to results
            for result, entities in zip(results, entities_list):
                for entity in entities:
                    result.add_entity(
                        text=entity["text"],
                        entity_type=entity["type"],
                        start_pos=entity["start_pos"],
                        end_pos=entity["end_pos"]
                    )
                    
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            # Don't fail the entire search if entity extraction fails
            
    def get_name(self) -> str:
        """Get the name of the search engine."""
        return "Google"
        
    def __str__(self) -> str:
        """String representation of the search engine."""
        return f"{self.__class__.__name__}({self.get_name()})"
    
    def __repr__(self) -> str:
        """Official string representation."""
        return f"<{self.__class__.__name__} name='{self.get_name()}'>"
    
    def search(
        self, 
        query: str, 
        max_results: int = 10,
        intent: Optional[SearchIntent] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search the web synchronously.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-100)
            intent: Optional search intent (person, location, thing)
            **kwargs: Additional search parameters
                - timeout: Override the default timeout
                - extract_entities: Whether to extract entities from results
                - search_engine: Override the default search engine
                
        Returns:
            List of search results
            
        Raises:
            ValueError: If max_results is not between 1 and 100
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.search_async(
                    query=query,
                    max_results=max_results,
                    intent=intent,
                    **kwargs
                )
            )
        finally:
            loop.close()
