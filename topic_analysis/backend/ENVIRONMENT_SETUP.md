# Backend Environment Configuration

## Supabase Configuration (Self-hosted on Hostinger/Coolify)

Create a `.env` file in the `backend` directory with the following configuration:

```bash
# Supabase Configuration (REQUIRED)
SUPABASE_URL=https://sbcontent.giniloh.com
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
SUPABASE_ANON_KEY=your-anon-key-here

# Database Configuration (Optional - for direct PostgreSQL connection if needed)
# Note: Most operations use Supabase SDK, but direct DB connection may be needed for migrations
DATABASE_URL=postgresql://postgres:your-password-here@sbcontent.giniloh.com:5432/postgres?sslmode=require

# Application Configuration
APP_NAME=Idea Burst API
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,https://yourdomain.com

# Redis (for caching and rate limiting)
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# External APIs (API keys are stored in Supabase api_keys table)
# Only non-API-key configuration here
AMAZON_ASSOCIATES_TAG=your-amazon-tag
WORDPRESS_API_URL=https://your-wordpress-site.com/wp-json

# File Uploads
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=text/csv,application/csv

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
```

## Important Notes

1. **Supabase URL**: `https://sbcontent.giniloh.com` - Your self-hosted Supabase instance
2. **Service Role Key**: Use this for admin operations (keep it secret!)
3. **Anon Key**: Safe to use in frontend, but backend can use it too
4. **Database URL**: Direct PostgreSQL connection string (if needed for migrations or direct SQL)
5. **Port**: PostgreSQL is on port `5432` (standard port)

## Environment Variables from Coolify

These are the key variables from your Coolify setup:

- `SERVICE_FQDN_SUPABASEKONG=sbcontent.giniloh.com`
- `SERVICE_URL_SUPABASEKONG=https://sbcontent.giniloh.com`
- `SERVICE_PASSWORD_POSTGRES=your-postgres-password-here`
- `SERVICE_PASSWORD_JWT=your-jwt-secret-here`
- `SERVICE_SUPABASEANON_KEY` (used as SUPABASE_ANON_KEY)
- `SERVICE_SUPABASESERVICE_KEY` (used as SUPABASE_SERVICE_ROLE_KEY)

## Verification

After setting up your `.env` file:

1. **Test Supabase connection**:
   ```bash
   python -c "from src.core.config import settings; print('Supabase URL:', settings.supabase_url)"
   ```

2. **Check environment variables are loaded**:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('SUPABASE_URL:', os.getenv('SUPABASE_URL'))"
   ```

3. **Start the backend**:
   ```bash
   python main.py
   ```

## Security Reminders

- Never commit `.env` files to version control
- Keep your service role key secure - it has admin access
- The anon key is safe for frontend use
- Rotate keys periodically

