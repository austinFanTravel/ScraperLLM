#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if containers are running
echo "🔍 Checking if containers are running..."
if ! docker-compose ps | grep -q "Up"; then
  echo "❌ Some containers are not running"
  docker-compose ps
  exit 1
fi

# Check API health
echo "🩺 Checking API health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${NGINX_PORT:-8000}/api/health)

if [ "$response" -eq 200 ]; then
  echo "✅ API is healthy (Status: $response)"
  exit 0
else
  echo "❌ API is not healthy (Status: $response)"
  echo "📋 Debug info:"
  docker-compose logs --tail=20
  exit 1
fi
