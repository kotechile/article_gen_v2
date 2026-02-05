# =============================================================================
# Profit Path - Frontend Dockerfile
# =============================================================================
# Multi-stage Docker build for Next.js application
# 
# Build: docker build -t profit-path-frontend .
# Run: docker run -p 3000:3000 profit-path-frontend
# =============================================================================

# =============================================================================
# STAGE 1: Base Image
# =============================================================================
FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; \
    else echo "Lockfile not found." && exit 1; \
    fi

# =============================================================================
# STAGE 2: Builder
# =============================================================================
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js collects completely anonymous telemetry data about general usage.
# Learn more here: https://nextjs.org/telemetry
# Uncomment the following line in case you want to disable telemetry during the build.
ENV NEXT_TELEMETRY_DISABLED=1

# Build time variables
# Build time variables - support both standard Next.js and VITE (Coolify) args
ARG NEXT_PUBLIC_API_KEY
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY

# VITE variants (often injected by Coolify/CRA-based tools)
ARG VITE_API_BASE_URL
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ARG VITE_DEBUG

# Generate .env.production for the build process
# We use shell expansion in RUN because Docker ENV treats empty strings as "set", prohibiting fallback.
RUN echo "NEXT_PUBLIC_API_KEY=${NEXT_PUBLIC_API_KEY}" >> .env.production && \
    echo "NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL:-$VITE_API_BASE_URL}" >> .env.production && \
    echo "NEXT_PUBLIC_SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL:-$VITE_SUPABASE_URL}" >> .env.production && \
    echo "NEXT_PUBLIC_SUPABASE_ANON_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY:-$VITE_SUPABASE_ANON_KEY}" >> .env.production && \
    echo "NEXT_PUBLIC_DEBUG=${NEXT_PUBLIC_DEBUG:-$VITE_DEBUG}" >> .env.production

# Debug: Print generated .env file (Warning: might show secrets in logs if not careful, but needed for debug)
RUN echo "--- Generated .env.production content ---" && cat .env.production

RUN npm run build
RUN echo "Checking .next directory contents:" && ls -la .next

# =============================================================================
# STAGE 3: Runner
# =============================================================================
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
# set hostname to localhost
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
