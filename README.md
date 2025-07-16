# ScraperLLM

A powerful web scraping and information extraction tool with built-in Named Entity Recognition (NER) and advanced search capabilities. ScraperLLM combines semantic search, keyword search, and social media integration to provide comprehensive search results.

## Features

- **Hybrid Search**: Combines semantic and keyword search for better results
- **Social Media Integration**: Search across multiple platforms (Twitter, Facebook, Instagram, TikTok)
- **Entity Extraction**: Built-in NER to identify people, organizations, locations, and more
- **Asynchronous Processing**: High-performance async/await support for efficient searching
- **Docker Support**: Easy deployment using Docker containers
- **RESTful API**: Modern FastAPI-based web interface with Swagger documentation
- **Semantic Search**: Advanced vector-based search capabilities
- **Customizable**: Configure search parameters and result filtering
- **Modular Architecture**: Easy to extend with custom search providers and processors

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- SerpAPI key (for web search functionality)
- (Optional) GPU for LLM acceleration

## Quick Start

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ScraperLLM.git
   cd ScraperLLM
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the development server:
   ```bash
   python run_web.py
   ```
   Access the API documentation at: http://localhost:8000/docs

### Docker Deployment

1. Build and start the containers:
   ```bash
   docker-compose up --build -d
   ```

2. The application will be available at: http://localhost:8000

3. To stop the containers:
   ```bash
   docker-compose down
   ```

## Configuration

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Key configuration options in `.env`:
- `SERPAPI_KEY`: Your SerpAPI key (required for web search)
- `MODEL_NAME`: Default model for semantic search (default: 'all-MiniLM-L6-v2')
- `LOG_LEVEL`: Logging level (default: 'INFO')
- `DEBUG`: Enable debug mode (default: False in production)

## Project Structure

```
scraper_llm/
├── core/               # Core functionality and configurations
├── search/             # Search providers and utilities
│   ├── __init__.py
│   ├── base.py         # Base classes for search engines
│   ├── serpapi_search.py  # SerpAPI search implementation
│   └── semantic_search.py # Semantic search implementation
├── utils/              # Utility functions
│   ├── content_processor.py  # Web content processing
│   └── semantic_search_utils.py  # Semantic search utilities
├── web/                # Web interface (FastAPI)
│   ├── app.py          # FastAPI application
│   └── templates/      # HTML templates (if any)
├── .dockerignore
├── .env.example        # Example environment variables
├── .gitignore
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── pyproject.toml      # Project metadata and dependencies
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control. Use environment variables or a secure secret management system.
2. **Input Validation**: Always validate and sanitize user inputs to prevent injection attacks.
3. **Rate Limiting**: Consider implementing rate limiting for production deployments.
4. **HTTPS**: Always use HTTPS in production to encrypt data in transit.
5. **Docker Security**: Run containers as non-root users and keep the host system updated.

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

3. Code style and quality:
   ```bash
   black .
   isort .
   flake8
   mypy .
   ```

### Adding a New Search Provider

1. Create a new file in `scraper_llm/search/`
2. Implement the `SearchEngine` interface from `scraper_llm.search.base`
3. Update the search factory to include your new provider

## Deployment

### AWS EC2 Deployment

1. Launch an EC2 instance (Ubuntu 22.04 recommended)
2. Install Docker and Docker Compose
3. Clone the repository
4. Configure environment variables
5. Start the containers:
   ```bash
   docker-compose up -d --build
   ```

### Production Considerations

- Use a reverse proxy (Nginx/Apache) with HTTPS
- Set up proper logging and monitoring
- Implement backup strategies
- Use environment-specific configurations
- Monitor resource usage and scale as needed

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
