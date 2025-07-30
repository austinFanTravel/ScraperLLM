#!/bin/bash

# Build the minimal Docker image
echo "Building minimal Docker image..."
docker build -t scraperllm-minimal -f Dockerfile.minimal .

# Check the image size
echo -e "\nImage size:"
docker images scraperllm-minimal --format "{{.Size}}"

# Run a test container to verify it works
echo -e "\nTesting the container..."
docker run --rm -d -p 8000:8000 --name scraperllm-test scraperllm-minimal

# Check if the container is running
if docker ps | grep -q scraperllm-test; then
    echo -e "\nContainer is running! Testing API..."
    sleep 2  # Give the server time to start
    
    # Test the health endpoint
    curl -s http://localhost:8000/health | jq .
    
    # Stop the test container
    echo -e "\nStopping test container..."
    docker stop scraperllm-test
else
    echo "Failed to start container. Check the logs with: docker logs scraperllm-test"
fi
