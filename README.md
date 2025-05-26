# ScraperLLM

A powerful web scraping and information extraction tool with built-in Named Entity Recognition (NER) capabilities. ScraperLLM allows you to search the web, extract structured information, and analyze text content with ease.

## Features

- **Web Search**: Search across multiple search engines (Google, Bing, etc.)
- **Entity Extraction**: Built-in NER to identify people, organizations, locations, and more
- **Asynchronous Processing**: High-performance async/await support for efficient scraping
- **Configurable**: Easy to configure with environment variables and settings
- **Extensible**: Plugin architecture for adding custom search engines and entity extractors

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

## Usage

### Basic Search

```bash
python -m scraper_llm search "Python programming"
```

### Search with Entity Extraction

```bash
python -m scraper_llm search "Elon Musk" --extract-entities
```

### Save Results to File

```bash
python -m scraper_llm search "Latest AI research" --output results.json
```

### CLI Options

```
Usage: python -m scraper_llm search [OPTIONS] QUERY

  Search the web and extract information.

  Examples:

      $ scraper-llm search "Python programming"
      $ scraper-llm search "Elon Musk" --max-results 5 --output results.json
      $ scraper-llm search "Latest AI research" --extract-entities

Arguments:
  QUERY  Search query  [required]

Options:
  -m, --max-results INTEGER  Maximum number of results to return.  [default: 50]
  -o, --output FILE          Output file to save results (JSON format).
  -e, --extract-entities     Extract named entities from search results.
  --help                     Show this message and exit.
```

## Configuration

Create a `.env` file in the project root to override default settings:

```ini
# Application settings
DEBUG=True
LOG_LEVEL=INFO

# File paths
DATA_DIR=./data
LOGS_DIR=./logs

# Search settings
SEARCH_TIMEOUT=30
MAX_RESULTS=50

# NER settings
NER_MODEL=en_core_web_sm
```

## Development

### Running Tests

```bash
pytest
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

### Linting

```bash
flake8
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
