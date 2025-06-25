FROM python:3.10-slim

WORKDIR /code

# Install system dependencies required for cartopy, ML libraries, and multimedia
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgeos-c1v5 \
    libgdal-dev \
    gdal-bin \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create data directory and set permissions
RUN mkdir -p /code/data && \
    chmod 777 /code/data

# Create matplotlib config directory and set permissions
RUN mkdir -p /tmp/matplotlib && \
    chmod 777 /tmp/matplotlib

# Create cache directories for ML libraries
RUN mkdir -p /tmp/.cache && \
    chmod 777 /tmp/.cache

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV DATA_PATH=/code/data
ENV DASH_DEBUG_MODE=false
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV NUMBA_CACHE_DIR=/tmp/.cache/numba
ENV UMAP_DISABLE_CHECK_RANDOM_STATE=1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir protobuf>=3.20.0,<4.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /code && \
    chown -R appuser:appuser /tmp

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start the application with optimized settings
CMD ["python", "app.py"]