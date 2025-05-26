"""Tests for the search functionality."""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import HttpUrl

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from pydantic import HttpUrl

from scraper_llm.search import (
    SearchEngine,
    SearchResult,
    SearchIntent,
    EntityType,
    Entity,
    WebSearcher,
)


# Fixtures
@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Sample search results for testing."""
    return [
        SearchResult(
            title="Test Result 1",
            url=HttpUrl("https://example.com/1"),
            snippet="This is a test result with some sample text.",
            source="test",
            entities=[
                {"text": "Test", "type": EntityType.OTHER, "start_pos": 0, "end_pos": 4}
            ],
        ),
        SearchResult(
            title="Test Result 2",
            url=HttpUrl("https://example.com/2"),
            snippet="Another test result with entities like New York and Python.",
            source="test",
            entities=[
                {"text": "New York", "type": EntityType.LOCATION, "start_pos": 30, "end_pos": 38},
                {"text": "Python", "type": EntityType.OTHER, "start_pos": 44, "end_pos": 50},
            ],
        ),
    ]


# Tests
class TestSearchResult:
    """Tests for the SearchResult class."""
    
    def test_to_dict(self, sample_search_results):
        """Test converting SearchResult to dictionary."""
        result = sample_search_results[0]
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["title"] == result.title
        assert result_dict["url"] == str(result.url)
        assert result_dict["snippet"] == result.snippet
        assert result_dict["source"] == result.source
        assert "entities" in result_dict
        assert len(result_dict["entities"]) == len(result.entities)
    
    def test_add_entity(self):
        """Test adding entities to a search result."""
        result = SearchResult(
            title="Test",
            url=HttpUrl("https://example.com"),
            snippet="Test snippet"
        )
        
        # Add an entity
        result.add_entity("Python", EntityType.OTHER, 10, 16)
        
        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity["text"] == "Python"
        assert entity["type"] == EntityType.OTHER
        assert entity["start_pos"] == 10
        assert entity["end_pos"] == 16


class TestWebSearcher:
    """Tests for the WebSearcher class."""
    
    @pytest.mark.asyncio
    async def test_search_async(self, mock_web_searcher, sample_search_results):
        """Test async search functionality."""
        # Setup mock
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text.return_value = '<html><div class="g"><h3>Test Result 1</h3><a href="https://example.com/1">Link</a><div class="IsZvec">Snippet 1</div></div></html>'
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test search
            searcher = WebSearcher()
            results = await searcher.search_async("test query", max_results=2)
            
            # Verify results
            assert len(results) == 2
            assert results[0].title == "Test Result 1"
            assert results[1].title == "Test Result 2"
    
    @pytest.mark.asyncio
    async def test_search_with_intent(self, sample_search_results):
        """Test search with intent parameter."""
        # Setup mock
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text.return_value = '<html></html>'
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test search with intent
            searcher = WebSearcher()
            results = await searcher.search_async(
                "test query", 
                max_results=2, 
                intent=SearchIntent.PERSON
            )
            
            # Verify results
            assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_extract_entities_async(self, sample_search_results):
        """Test async entity extraction."""
        # Setup mock
        searcher = WebSearcher()
        searcher.extract_entities = True
        
        with patch.object(searcher, '_search_google', return_value=sample_search_results):
            results = await searcher.search_async("test query", max_results=2)
            
            # Verify entities were extracted
            assert len(results) > 0
            assert hasattr(results[0], 'entities')


class TestSearchEngine:
    """Tests for the base SearchEngine class."""
    
    def test_search_sync(self):
        """Test synchronous search wrapper."""
        class TestEngine(SearchEngine):
            async def search_async(self, *args, **kwargs):
                return [SearchResult("Test Result", "https://example.com", "Test snippet")]
                
            def get_name(self):
                return "TestEngine"
                
        engine = TestEngine()
        results = engine.search("test query", max_results=2)
        assert len(results) == 1
        assert results[0].title == "Test Result"
    
    def test_get_name_raises_not_implemented(self):
        """Test that get_name raises NotImplementedError if not overridden."""
        class TestEngine(SearchEngine):
            async def search_async(self, *args, **kwargs):
                return []
                
        engine = TestEngine()
        with pytest.raises(NotImplementedError):
            engine.get_name()


# Test the CLI integration
class TestCLI:
    """Tests for the CLI interface."""
    
    @patch('scraper_llm.core.logging.configure_logging')
    def test_cli_help(self, mock_logging, runner):
        """Test the CLI help output."""
        from scraper_llm.cli import app
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Search the web and return results" in result.output
    
    @patch('scraper_llm.core.logging.configure_logging')
    def test_search_command(self, mock_logging, runner, sample_search_results):
        """Test the search command."""
        from scraper_llm.cli import app
        
        # Setup mock
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text.return_value = '<html><div class="g"><h3>Test Result 1</h3><a href="https://example.com/1">Link</a><div class="IsZvec">Snippet 1</div></div></html>'
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Run the command
            result = runner.invoke(app, ["search", "test query"])
            
            # Verify output
            assert result.exit_code == 0
            assert "Test Result 1" in result.output
    
    @patch('scraper_llm.core.logging.configure_logging')
    def test_search_with_output_file(self, mock_logging, runner, sample_search_results, tmp_path):
        """Test search with output to file."""
        from scraper_llm.cli import app
        
        # Setup mock
        output_file = tmp_path / "results.json"
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text.return_value = '<html><div class="g"><h3>Test Result 1</h3><a href="https://example.com/1">Link</a><div class="IsZvec">Snippet 1</div></div></html>'
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Run the command with output file
            result = runner.invoke(app, ["search", "test query", "--output", str(output_file)])
            
            # Verify output file was created
            assert result.exit_code == 0
            assert output_file.exists()
            assert "Test Result 1" in output_file.read_text()


# Test the NER functionality
class TestNER:
    """Tests for the NER functionality."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_async(self):
        """Test async entity extraction."""
        from scraper_llm.utils.ner import extract_entities_async
        
        with patch('nltk.ne_chunk') as mock_ne_chunk, \
             patch('nltk.pos_tag', return_value=[('Google', 'NNP'), ('is', 'VBZ'), 
                   ('based', 'VBN'), ('in', 'IN'), ('Mountain', 'NNP'), 
                   ('View', 'NNP'), (',', ','), ('California', 'NNP')]):
            
            # Mock the named entity recognition
            mock_ne_chunk.return_value = [
                ('ORGANIZATION', 'Google'), 
                ('LOCATION', 'Mountain View'),
                ('LOCATION', 'California')
            ]
            
            # Test with a simple sentence
            texts = ["Google is based in Mountain View, California"]
            entities_list = await extract_entities_async(texts)
            
            # Should find Google (ORG) and Mountain View (LOC)
            assert len(entities_list) == 1
            entities = entities_list[0]
            assert len(entities) >= 2
        
    def test_extract_entities_sync(self):
        """Test synchronous entity extraction."""
        from scraper_llm.utils.ner import extract_entities
        
        with patch('nltk.ne_chunk') as mock_ne_chunk, \
             patch('nltk.pos_tag', return_value=[('Google', 'NNP'), ('is', 'VBZ'), 
                   ('based', 'VBN'), ('in', 'IN'), ('Mountain', 'NNP'), 
                   ('View', 'NNP'), (',', ','), ('California', 'NNP')]):
            
            # Mock the named entity recognition
            mock_ne_chunk.return_value = [
                ('ORGANIZATION', 'Google'), 
                ('LOCATION', 'Mountain View'),
                ('LOCATION', 'California')
            ]
            
            # Test with a simple sentence
            texts = ["Google is based in Mountain View, California"]
            entities_list = extract_entities(texts)
            
            # Should find Google (ORG) and Mountain View (LOC)
            assert len(entities_list) == 1
            entities = entities_list[0]
            assert len(entities) >= 2
        
        # Check that we found Google as an organization
        orgs = [e for e in entities if e["type"] == EntityType.ORGANIZATION]
        assert any(e["text"] == "Google" for e in orgs)
