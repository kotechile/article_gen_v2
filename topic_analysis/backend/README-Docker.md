# Trend Analysis Platform - Docker Guide

This guide explains how to use Docker with the Trend Analysis Platform backend.

## Quick Start

### Using Docker Compose (Recommended)

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f backend
   ```

3. **Stop services:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t trend-analysis-backend .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 trend-analysis-backend
   ```

## Available Build Targets

The Dockerfile supports multiple build targets:

- **`production`** (default): Production-ready image
- **`development`**: Development image with hot reload
- **`security-hardened`**: Production image with additional security tools
- **`minimal`**: Minimal Alpine-based image

### Build specific target:
```bash
docker build --target production -t trend-analysis-backend .
docker build --target development -t trend-analysis-dev .
docker build --target security-hardened -t trend-analysis-secure .
docker build --target minimal -t trend-analysis-minimal .
```

## Environment Variables

### Required Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key

### Optional Variables
- `SENDGRID_API_KEY`: Email service API key
- `GOOGLE_TRENDS_API_KEY`: Google Trends API key
- `OPENAI_API_KEY`: OpenAI API key
- `ENVIRONMENT`: Environment (development/staging/production)
- `DEBUG`: Enable debug mode (true/false)

## Services

### Backend API
- **Port:** 8000
- **Health Check:** `GET /health`
- **API Docs:** `GET /docs` (development only)

### PostgreSQL Database
- **Port:** 5432
- **Database:** trend_analysis
- **User:** postgres
- **Password:** password

### Redis Cache
- **Port:** 6379
- **Database:** 0

### Nginx (Optional)
- **Port:** 80 (HTTP), 443 (HTTPS)
- **Profile:** production

### Monitoring (Optional)
- **Prometheus:** Port 9090
- **Grafana:** Port 3000
- **Profile:** monitoring

## Development

### Start development environment:
```bash
docker-compose --profile development up -d
```

### Run tests:
```bash
docker-compose exec backend pytest
```

### Access database:
```bash
docker-compose exec postgres psql -U postgres -d trend_analysis
```

### View Redis:
```bash
docker-compose exec redis redis-cli
```

## Production Deployment

### Start production environment:
```bash
docker-compose --profile production up -d
```

### Start with monitoring:
```bash
docker-compose --profile production --profile monitoring up -d
```

## Security Features

- Non-root user execution
- Security headers middleware
- Rate limiting
- Request logging
- Health checks
- Security scanning tools (hardened mode)

## Troubleshooting

### View container logs:
```bash
docker-compose logs [service-name]
```

### Access container shell:
```bash
docker-compose exec backend /bin/bash
```

### Restart specific service:
```bash
docker-compose restart backend
```

### Clean up:
```bash
docker-compose down -v  # Remove volumes
docker system prune -a  # Clean up all unused images
```

## Performance Tuning

### For production:
- Use `security-hardened` target
- Set appropriate worker count
- Configure resource limits
- Use external database and Redis

### For development:
- Use `development` target
- Enable hot reload
- Mount source code as volume

## Monitoring

### Health Checks
- Backend: `GET /health`
- Database: `pg_isready`
- Redis: `redis-cli ping`

### Metrics
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

## Support

For issues and questions:
- Check container logs
- Verify environment variables
- Ensure all services are healthy
- Review Docker Compose configuration
