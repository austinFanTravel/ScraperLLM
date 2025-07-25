# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only the requirements files first for better caching
COPY pyproject.toml poetry.lock* ./
COPY requirements.txt ./

# Install Python dependencies from all requirements files
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN if [ -f requirements-web.txt ]; then pip install --no-cache-dir -r requirements-web.txt; fi
RUN if [ -f requirements-llm.txt ]; then pip install --no-cache-dir -r requirements-llm.txt; fi
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Install loguru explicitly since it's required
RUN pip install --no-cache-dir loguru

# Install spaCy and the English model directly
RUN pip install --no-cache-dir spacy && \
    python -m pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz && \
    python -c "import en_core_web_sm; en_core_web_sm.load()"

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "8000"]
