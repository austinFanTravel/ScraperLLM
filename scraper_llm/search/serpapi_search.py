"""SerpAPI implementation for web search."""
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from serpapi import GoogleSearch
from loguru import logger

from .base import SearchEngine, SearchResult, SearchIntent, Entity, EntityType
from ..core.config import get_settings


class SerpAPISearcher(SearchEngine):
    """Search engine implementation using SerpAPI for Google search results."""
    
    def __init__(self, api_key: Optional[str] = None, extract_entities: bool = False):
        """Initialize the SerpAPI searcher.
        
        Args:
            api_key: SerpAPI API key. If not provided, will try to get from settings.
            extract_entities: Whether to extract entities from search results.
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.SERPAPI_KEY
        
        if not self.api_key:
            raise ValueError(
                "SerpAPI key not provided and not found in settings. "
                "Please set the SERPAPI_KEY environment variable or pass it directly."
            )
            
        self.extract_entities = extract_entities
        self.logger = logger.bind(searcher="SerpAPI")
    
    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        intent: Optional[SearchIntent] = None,
        sites: Optional[List[str]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Perform an asynchronous search using SerpAPI.
        
        Args:
            query: The search query string.
            max_results: Maximum number of results to return (1-100).
            intent: Optional search intent to filter results.
            sites: Optional list of domains to limit the search to.
                 Example: ["example.com", "test.org"] will only return results from these domains.
            **kwargs: Additional search parameters.
                - location: Geographic location for search
                - tbs: Time-based search filter (e.g., 'qdr:h' for past hour)
                - any other SerpAPI parameters
                
        Returns:
            List of search results.
        """
        try:
            # Build the search query with site restrictions if provided
            search_query = query
            if sites and isinstance(sites, list):
                # Format: (query) (site:example1.com OR site:example2.com)
                site_restrictions = " OR ".join(f"site:{site.strip()}" for site in sites if site.strip())
                if site_restrictions:
                    search_query = f"({query}) ({site_restrictions})"
            
            # Prepare search parameters
            params = {
                "q": search_query,  # Use the modified query with site restrictions
                "num": min(max(1, max_results), 100),  # Ensure between 1-100
                "api_key": self.api_key,
                "hl": "en",  # Language: English
                "gl": "us",  # Country: United States
            }
            
            # Add any additional parameters from kwargs
            if "location" in kwargs:
                params["location"] = kwargs["location"]
            if "tbs" in kwargs:  # Time-based search (e.g., 'qdr:h' for past hour)
                params["tbs"] = kwargs["tbs"]
            
            # Execute the search
            search = GoogleSearch(params)
            # Search within specific sites
            results = await searcher.search_async(
                query,
                sites=["google.com", "instagram.com", "twitter.com", "facebook.com", "youtube.com", "tmz.com", "tiktok.com"], # Sites to search within **Implement Reddit here once search function shows results appropriately
                max_results=max_results,
                **kwargs
            )
            
            # Parse the results
            search_results = []
            
            # Handle organic results
            if "organic_results" in results:
                for result in results["organic_results"][:max_results]:
                    try:
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("link", ""),
                            snippet=result.get("snippet", ""),
                            source="serpapi"
                        )
                        search_results.append(search_result)
                    except Exception as e:
                        self.logger.error(f"Error parsing result: {e}")
                        continue
            
            # Rest of the method remains the same...
            # [Previous code for handling answer boxes, knowledge graph, etc.]
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
            
            # If no organic results, check for answer box or knowledge graph
            if not search_results and ("answer_box" in results or "knowledge_graph" in results):
                if "answer_box" in results and "answer" in results["answer_box"]:
                    answer = results["answer_box"]["answer"]
                    search_results.append(SearchResult(
                        title=query,
                        url=results.get("search_metadata", {}).get("google_url", ""),
                        snippet=answer,
                        source="serpapi_answer_box"
                    ))
                elif "knowledge_graph" in results:
                    kg = results["knowledge_graph"]
                    search_results.append(SearchResult(
                        title=kg.get("title", query),
                        url=kg.get("source", {}).get("link", ""),
                        snippet=kg.get("description", ""),
                        source="serpapi_knowledge_graph"
                    ))
            
            # If we have an intent, filter results
            if intent:
                search_results = await self._filter_by_intent(search_results, intent)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    async def _filter_by_intent(
        self, 
        results: List[SearchResult], 
        intent: SearchIntent
    ) -> List[SearchResult]:
        """Filter search results by intent.
        
        Args:
            results: List of search results to filter.
            intent: The search intent to filter by.
            
        Returns:
            Filtered list of search results.
        """
        # For now, we'll just return all results
        # In a real implementation, you might want to filter based on the intent
        # For example, for PERSON intent, you might want to prioritize results
        # that contain personal information
        return results

    def get_name(self) -> str:
        """Get the name of the search engine."""
        return "SerpAPI"


# For backward compatibility
WebSearcher = SerpAPISearcher
