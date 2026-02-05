ARG COOLIFY_FIX=true

# =============================================================================
# Trend Analysis Platform - Nginx Dockerfile
# =============================================================================
FROM nginx:alpine AS base

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom config
COPY trend-analysis.conf /etc/nginx/conf.d/default.conf

# =============================================================================
# Final Stage
# =============================================================================
FROM base AS release
