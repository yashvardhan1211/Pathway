#!/bin/bash

# Run script for dockerized stock predictor

echo "Starting dockerized stock predictor..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo "See DOCKER_SETUP.md for detailed instructions."
    exit 1
fi

# Try docker compose (newer) first, then docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "Using docker compose..."
    docker compose up -d
    echo "Container started successfully!"
    echo "Test Pathway functionality with: docker compose exec stock-predictor python main.py demo"
    echo "Access logs with: docker compose logs -f"
    echo "Stop container with: docker compose down"
    echo "Access container shell with: docker compose exec stock-predictor bash"
elif command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose up -d
    echo "Container started successfully!"
    echo "Test Pathway functionality with: docker-compose exec stock-predictor python main.py demo"
    echo "Access logs with: docker-compose logs -f"
    echo "Stop container with: docker-compose down"
    echo "Access container shell with: docker-compose exec stock-predictor bash"
else
    echo "ERROR: Neither 'docker compose' nor 'docker-compose' is available!"
    echo "Please ensure Docker Desktop is properly installed and running."
    exit 1
fi