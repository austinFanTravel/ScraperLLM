#!/bin/bash
set -e

echo "🚀 Starting ScraperLLM deployment..."

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Error: .env file not found"
    exit 1
fi

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p nginx/ssl/your-domain.com
echo "✅ Created nginx/ssl/your-domain.com"

mkdir -p data/search_assistant
echo "✅ Created data/search_assistant"

mkdir -p data/logs/nginx
echo "✅ Created data/logs/nginx"

# Set permissions
echo "🔒 Setting permissions..."
chmod -R 755 data
chmod 600 .env 2>/dev/null || true

# Build and start containers
echo "🐳 Starting Docker containers..."
docker-compose down
docker-compose up -d --build

# Show logs
echo "📝 Showing logs..."
docker-compose logs -f --tail=50

echo "✅ Deployment complete! Your ScraperLLM instance is now running."
echo "🌐 Access it at: http://${NGINX_HOST}:${NGINX_PORT}"
