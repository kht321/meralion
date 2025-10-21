# MERaLiON ASR Evaluation Toolkit - Docker Container
# Optimized for reproducible ASR evaluation environments

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg: Required for audio processing (librosa, pydub)
# - git: For version control operations
# - build-essential: For compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface

# Create cache directories
RUN mkdir -p /app/.cache/torch /app/.cache/huggingface

# Default command: start a bash shell
CMD ["/bin/bash"]
