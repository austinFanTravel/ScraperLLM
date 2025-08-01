# ScraperLLM API

A FastAPI-based web service that provides web scraping and semantic query expansion capabilities.

## 🚀 Features

- **Web Scraping**: Extract content from web pages using CSS selectors
- **Query Expansion**: Generate semantically related search queries using a trained AI model
- **RESTful API**: Well-documented endpoints with OpenAPI/Swagger UI
- **CORS Support**: Pre-configured for Webflow integration
- **Docker Support**: Easy containerization for deployment
- **Railway Ready**: Optimized for deployment on Railway.app

## 🏗️ Project Structure

```
scraperllm/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Application configuration
│   │   └── logging.py       # Logging configuration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── status.py        # Health check and system status
│   │   ├── scrape.py        # Web scraping endpoints
│   │   └── query.py         # Query expansion endpoints
│   └── services/
│       ├── __init__.py
│       └── semantic_expander.py  # AI model integration
├── models/                  # Trained model files (not in version control)
├── tests/                   # Test files
├── .env.example             # Example environment variables
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── railway.toml             # Railway.app deployment config
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Docker (for containerized deployment)
- A trained model file in the `models/` directory

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ScraperLLM.git
   cd ScraperLLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.

5. Run the development server:
   ```bash
   uvicorn api.main:app --reload
   ```

6. Access the API documentation at `http://localhost:8000/docs`

## 🛠️ API Endpoints

### Status
- `GET /api/status` - Check API status and system information
- `GET /api/health` - Simple health check endpoint

### Scraping
- `POST /api/scrape` - Scrape a website with CSS selectors

### Query Expansion
- `POST /api/query/expand` - Generate semantically related search queries
- `GET /api/query/status` - Check the status of the semantic expansion model

## 🐳 Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t scraperllm-api .
   ```

2. Run the container:
   ```bash
   docker run -d --name scraperllm-api -p 8000:8000 --env-file .env scraperllm-api
   ```

## 🚂 Railway Deployment

1. Install the Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

3. Link your project:
   ```bash
   railway link
   ```

4. Set environment variables:
   ```bash
   railway env push .env
   ```

5. Deploy:
   ```bash
   railway up
   ```

## 🔧 Configuration

Environment variables can be set in the `.env` file or in your deployment platform's environment settings:

- `MODEL_PATH`: Path to the trained model directory (default: `models/query_expander`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: `http://localhost:3000,https://*.webflow.io,https://livewebscraper.com`)
- `RATE_LIMIT`: Maximum requests per minute (default: `60`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - For HTML parsing
- [Hugging Face](https://huggingface.co/) - For the transformer models
- [Railway](https://railway.app/) - For deployment hosting
