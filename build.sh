#!/bin/bash

# Build script for dockerized stock predictor

echo "Building Docker image for stock predictor..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo "See DOCKER_SETUP.md for detailed instructions."
    exit 1
fi

# Create necessary directories
mkdir -p data output models src

# Try docker compose (newer) first, then docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "Using docker compose..."
    docker compose build
elif command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose build
else
    echo "ERROR: Neither 'docker compose' nor 'docker-compose' is available!"
    echo "Please ensure Docker Desktop is properly installed and running."
    exit 1
fi

echo "Docker image built successfully!"
echo "To run the container, use: ./run.sh"