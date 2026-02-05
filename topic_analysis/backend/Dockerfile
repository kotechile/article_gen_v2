# =============================================================================
# Trend Analysis Platform - Backend Dockerfile
# =============================================================================
# Multi-stage Docker build for production-ready FastAPI backend
# 
# Build: docker build -t trend-analysis-backend .
# Run: docker run -p 8000:8000 trend-analysis-backend
# =============================================================================

# =============================================================================
# STAGE 1: Base Image with Python and System Dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PostgreSQL client libraries
    libpq-dev \
    # Build tools for compiling Python packages
    gcc \
    g++ \
    make \
    # Security updates
    ca-certificates \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# STAGE 2: Dependencies Installation
# =============================================================================
FROM base as dependencies

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# STAGE 3: Development Stage (Optional)
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# STAGE 4: Production Stage
# =============================================================================
FROM dependencies as production

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# STAGE 5: Security Hardened Production Stage
# =============================================================================
FROM production as security-hardened

# Install additional security tools
USER root
RUN apt-get update && apt-get install -y \
    # Security scanning tools
    clamav \
    # Network security tools
    netcat-openbsd \
    # Process monitoring
    htop \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create security directories
RUN mkdir -p /app/security /app/audit

# Copy security configurations
COPY docker/security/ /app/security/

# Set security permissions
RUN chmod 600 /app/security/* \
    && chown -R appuser:appuser /app/security /app/audit

# Switch back to non-root user
USER appuser

# Security environment variables
ENV SECURITY_MODE=hardened \
    AUDIT_LOGGING=true \
    SECURITY_SCANNING=true

# Override command for security-hardened mode
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "warning"]

# =============================================================================
# STAGE 6: Minimal Production Stage (Debian-based for better pandas support)
# =============================================================================
FROM python:3.11-slim as minimal

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PostgreSQL client libraries
    libpq-dev \
    # Build tools for compiling Python packages
    gcc \
    g++ \
    make \
    # Security updates
    ca-certificates \
    # Network tools
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# =============================================================================
# METADATA
# =============================================================================
LABEL maintainer="Trend Analysis Platform Team" \
      version="1.0.0" \
      description="Trend Analysis Platform Backend API" \
      org.opencontainers.image.title="Trend Analysis Platform Backend" \
      org.opencontainers.image.description="FastAPI backend for Trend Analysis Platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Trend Analysis Platform" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/trend-analysis/platform" \
      org.opencontainers.image.documentation="https://docs.trend-analysis.com" \
      org.opencontainers.image.created="2024-01-01T00:00:00Z" \
      org.opencontainers.image.revision="1.0.0"
