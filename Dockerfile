FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Node.js v20 and ffmpeg)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install --with-deps chromium

# Copy application code
COPY . .

# Install Remotion dependencies
RUN cd _remotion && npm install

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 5001

# Default command (can be overridden by docker-compose)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "600", "--workers", "2", "src.api.app:create_app()"]
