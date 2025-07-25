"""Base classes for search functionality."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, HttpUrl, validator


class SearchIntent(str, Enum):
    """Search intent types."""
    PERSON = "person"
    LOCATION = "location"
    THING = "thing"


class EntityType(str, Enum):
    """Named entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FACILITY"
    OTHER = "OTHER"


class Entity(TypedDict):
    """A named entity extracted from text."""
    text: str
    type: EntityType
    start_pos: int
    end_pos: int


class SearchResult(BaseModel):
    """Represents a single search result.
    
    Attributes:
        title: The title of the search result
        url: The URL of the result
        snippet: A brief summary or excerpt
        source: The search engine that provided this result
        entities: List of named entities found in the result
        timestamp: When the result was retrieved
        metadata: Additional metadata about the result
    """
    
    title: str = Field(..., description="The title of the search result")
    url: HttpUrl = Field(..., description="The URL of the result")
    snippet: str = Field(default="", description="A brief summary or excerpt")
    source: str = Field(default="web", description="The search engine that provided this result")
    entities: List[Entity] = Field(
        default_factory=list, 
        description="List of named entities found in the result"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the result was retrieved"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the result"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: str,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = self.dict()
        result["url"] = str(self.url)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @validator("url", pre=True)
    def validate_url(cls, v):
        """Ensure URLs are properly formatted."""
        if isinstance(v, str) and not v.startswith(('http://', 'https://')):
            return f"https://{v}"
        return v
    
    def add_entity(self, text: str, entity_type: EntityType, start_pos: int, end_pos: int) -> None:
        """Add a named entity to this result.
        
        Args:
            text: The entity text
            entity_type: Type of the entity
            start_pos: Start position in the text
            end_pos: End position in the text
        """
        self.entities.append({
            "text": text,
            "type": entity_type,
            "start_pos": start_pos,
            "end_pos": end_pos,
        })


class SearchEngine(ABC):
    """Abstract base class for search engines."""
    
    def __init__(self, **kwargs):
        """Initialize the search engine with optional configuration."""
        self.config = kwargs
        self.logger = kwargs.get("logger")
    
    @abstractmethod
    async def search_async(
        self, 
        query: str, 
        max_results: int = 10,
        intent: Optional[SearchIntent] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search asynchronously.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-100)
            intent: The search intent (person, location, thing)
            **kwargs: Additional search parameters
                - timeout: Maximum time to wait for search completion (seconds)
                - extract_entities: Whether to extract named entities
                
        Returns:
            List of search results
        """
        pass
    
    def search(
        self, 
        query: str, 
        max_results: int = 10,
        intent: Optional[SearchIntent] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search synchronously.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-100)
            intent: The search intent (person, location, thing)
            **kwargs: Additional search parameters
                - timeout: Maximum time to wait for search completion (seconds)
                - extract_entities: Whether to extract named entities
                
        Returns:
            List of search results
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.search_async(query, max_results, intent=intent, **kwargs)
        )
    
    async def extract_entities_async(
        self,
        texts: List[str],
        entity_types: Optional[List[EntityType]] = None
    ) -> List[List[Entity]]:
        """Extract named entities from text asynchronously.
        
        Args:
            texts: List of text strings to process
            entity_types: Optional list of entity types to extract
            
        Returns:
            List of lists of entities for each input text
        """
        # Default implementation uses NLTK
        from ..utils.ner import extract_entities_async
        return await extract_entities_async(texts)
    
    def extract_entities(
        self,
        texts: List[str],
        entity_types: Optional[List[EntityType]] = None
    ) -> List[List[Entity]]:
        """Extract named entities from text synchronously.
        
        Args:
            texts: List of text strings to process
            entity_types: Optional list of entity types to extract
            
        Returns:
            List of lists of entities for each input text
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.extract_entities_async(texts, entity_types=entity_types)
        )
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the search engine."""
        pass
    
    def __str__(self) -> str:
        """String representation of the search engine."""
        return f"{self.__class__.__name__}({self.get_name()})"
    
    def __repr__(self) -> str:
        """Official string representation."""
        return f"<{self.__class__.__name__} name='{self.get_name()}'>"
