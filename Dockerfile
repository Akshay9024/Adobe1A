# Multi-stage build for optimized image size
# Stage 1: Builder
FROM --platform=linux/amd64 python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create wheels directory
RUN mkdir /wheels

# Copy requirements
COPY requirements.txt /tmp/

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.9-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # PDF rendering
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # OCR support
    tesseract-ocr \
    tesseract-ocr-eng \
    # Additional language packs (optional)
    tesseract-ocr-jpn \
    tesseract-ocr-hin \
    tesseract-ocr-ara \
    tesseract-ocr-chi-sim \
    # Utilities
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Copy requirements to runtime stage
COPY requirements.txt /tmp/

# Install Python packages from wheels (faster, no compilation)
RUN pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY src/*.py ./src/

# Copy models directory if it exists
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=8
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Create non-root user for security
RUN useradd -m -u 1000 pdfuser && \
    chown -R pdfuser:pdfuser /app

# Switch to non-root user
USER pdfuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Entry point script
COPY --chown=pdfuser:pdfuser docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]