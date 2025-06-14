# Use PyTorch official image with CUDA 12.1 (compatible with your CUDA 12.9)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements_api.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the essential files
COPY api_multimodal.py .
COPY requirements_api.txt requirements.txt

# Create necessary directories
RUN mkdir -p load_test_results
RUN mkdir -p Three_models

# Expose the port
EXPOSE 4000

# Command to run the application
CMD ["python", "api_multimodal.py"]