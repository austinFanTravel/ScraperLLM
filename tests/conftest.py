"""Configuration and fixtures for pytest."""
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_imports():
    """
    Mock external dependencies to speed up tests.
    
    This prevents actual network calls and uses of heavy libraries
    during testing unless explicitly needed.
    """
    with patch('nltk.download', return_value=None), \
         patch('nltk.data.find', return_value=True), \
         patch('nltk.ne_chunk', return_value=[]), \
         patch('nltk.pos_tag', return_value=[('test', 'NN')]), \
         patch('requests.get', return_value=MagicMock(text='', status_code=200)), \
         patch('rich.console.Console.print'):
        yield


@pytest.fixture
def mock_web_searcher():
    """Mock WebSearcher for testing."""
    mock = MagicMock()
    mock.search_async = AsyncMock(return_value=[])
    mock.search = MagicMock(return_value=[])
    
    with patch('scraper_llm.search.web.WebSearcher', return_value=mock) as mock_class:
        yield mock_class.return_value
