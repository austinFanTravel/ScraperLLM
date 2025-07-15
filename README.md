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
- **Modular Architecture**: Easy to extend with custom search providers and processors

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ScraperLLM.git
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

## Configuration

Copy the example configuration file and update it with your settings:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:
- API keys for social media platforms
- Database connection settings
- Search parameters
- Logging preferences

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

## Project Structure

```
scraper_llm/
├── core/               # Core functionality
├── search/             # Search providers and utilities
├── models/             # Data models
├── utils/              # Utility functions
├── config.py           # Configuration settings
└── __main__.py         # Command-line interface
```

## Development

### Setting Up for Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Run linters:
   ```bash
   black .
   flake8
   mypy .
   ```

### Adding a New Search Provider

1. Create a new file in `scraper_llm/search/providers/`
2. Implement the `SearchProvider` interface
3. Register the provider in `scraper_llm/search/__init__.py`

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on the [GitHub repository](https://github.com/yourusername/ScraperLLM/issues).
