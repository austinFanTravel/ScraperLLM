# ScraperLLM

A powerful web scraping and information extraction tool with built-in Named Entity Recognition (NER) and advanced search capabilities. ScraperLLM combines semantic search, keyword search, and social media integration to provide comprehensive search results.

## Features

- **Hybrid Search**: Combines semantic and keyword search for better results
- **Social Media Integration**: Search across multiple platforms (Twitter, Facebook, Instagram, TikTok)
- **RSS Feed Support**: Monitor and search through custom RSS feeds
- **Entity Extraction**: Built-in NER to identify people, organizations, locations, and more
- **Asynchronous Processing**: High-performance async/await support for efficient searching
- **Search History**: Tracks and stores search history for future reference
- **Relevance Scoring**: Advanced algorithms to rank search results by relevance
- **Customizable**: Configure search parameters and result filtering

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ScraperLLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt  # For development
   ```

## Quick Start

### Basic Search

```python
from scraper_llm.search_assistant import SearchAssistant

# Initialize the search assistant
assistant = SearchAssistant()

# Perform a search
results = await assistant.search("Python programming", num_results=5)
```

### Search with Social Media

```python
# Search across social media platforms
results = await assistant.search("latest tech news", 
                              domains=["twitter.com", "facebook.com"])
```

### Add RSS Feeds

```python
# Add RSS feeds for specialized search
assistant.add_rss_feed("Tech News", "https://example.com/tech/feed")
assistant.add_rss_feed("Sports", "https://example.com/sports/feed")
```

## SearchAssistant API

### Initialization

```python
assistant = SearchAssistant(
    model_name="all-mpnet-base-v2",  # Default model
    data_dir="./data/search_assistant",  # Where to store data
    use_gpu=False  # Enable if you have CUDA
)
```

### Search Methods

```python
# Basic search
results = await assistant.search(
    query="your search query",
    num_results=10,         # Number of results to return
    min_relevance=0.2,      # Minimum relevance score (0.0-1.0)
    domains=None,           # Optional domain filter
    use_hybrid=True        # Use hybrid search (semantic + keyword)
)

# Add training examples for better results
assistant.add_training_example(
    query="machine learning",
    preferred_results=[
        {"text": "Introduction to Machine Learning"},
        {"text": "ML algorithms explained"}
    ],
    negative_results=[
        {"text": "General programming concepts"}
    ]
)
```

## Configuration

Create a `.env` file in the project root to configure settings:

```ini
# API Keys
SERPAPI_KEY=your_serpapi_key

# Search Settings
SEARCH_TIMEOUT=30
MAX_RESULTS=20
MIN_RELEVANCE=0.2

# Social Media Settings
SOCIAL_MEDIA_ENABLED=true
RSS_UPDATE_INTERVAL=3600  # seconds

# Paths
DATA_DIR=./data
CACHE_DIR=./cache
```

## Command Line Interface

```bash
# Basic search
python -m scraper_llm search "your query"

# Search with options
python -m scraper_llm search "tech news" \
    --max-results 15 \
    --min-relevance 0.3 \
    --domains twitter.com,facebook.com

# Add RSS feed
python -m scraper_llm add-feed "Tech News" https://example.com/feed/
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Support

For support, please open an issue in the GitHub repository.
