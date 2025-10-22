# Use Ubuntu 22.04 as base image for Linux environment
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip (if they don't exist)
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/models

# Copy application code
COPY . .

# Expose port for visualizations
EXPOSE 8080

# Set default command
CMD ["python", "main.py"]