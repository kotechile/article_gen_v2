# Supabase Integration Documentation

This document provides comprehensive information about the Supabase integration implementation in the Trend Analysis Platform backend.

## Overview

The Supabase integration replaces direct PostgreSQL connections with the Supabase SDK, providing:
- **Unified Database Access**: All database operations go through Supabase SDK
- **Real-time Features**: WebSocket-based real-time updates
- **Authentication**: Integrated user management
- **Scalability**: Cloud-hosted PostgreSQL with automatic scaling
- **Security**: Row Level Security (RLS) policies

## Architecture

### Core Components

1. **SupabaseClient** (`src/database/supabase_client.py`)
   - Manages Supabase client initialization
   - Handles connection status and health checks
   - Provides both service role and anonymous clients

2. **SupabaseService** (`src/services/supabase_service.py`)
   - CRUD operations using Supabase SDK
   - Operation logging and monitoring
   - Error handling and retry logic

3. **DatabaseOperationService** (`src/services/database_operation_service.py`)
   - Tracks all database operations
   - Provides operation statistics and monitoring
   - Handles operation cleanup and maintenance

4. **MigrationService** (`src/services/migration_service.py`)
   - Manages database migrations
   - Handles data migration from old PostgreSQL
   - Provides rollback capabilities

5. **RealTimeService** (`src/services/realtime_service.py`)
   - Manages real-time subscriptions
   - Handles WebSocket connections
   - Processes real-time events

6. **SupabaseErrorHandler** (`src/services/supabase_error_handler.py`)
   - Handles Supabase-specific errors
   - Translates errors to appropriate HTTP responses
   - Provides detailed error logging

7. **SupabaseConnectionManager** (`src/services/supabase_connection_manager.py`)
   - Manages connection pooling
   - Handles reconnection logic
   - Provides session management

## API Endpoints

### Health Check
- `GET /api/v1/health/database` - Check database connection health

### Database Operations
- `GET /api/v1/database/operations` - List all database operations
- `GET /api/v1/database/operations/{operation_id}` - Get specific operation details

### Migration Management
- `POST /api/v1/database/migrate` - Initiate database migration
- `GET /api/v1/database/migrate/{migration_id}` - Get migration status

### Real-time Features
- `POST /api/v1/database/realtime/subscribe` - Create real-time subscription
- `DELETE /api/v1/database/realtime/subscribe/{subscription_id}` - Remove subscription
- `WS /api/v1/ws/{channel_id}` - WebSocket connection for real-time updates

## Configuration

### Environment Variables

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key

# Optional: Connection settings
SUPABASE_CONNECTION_RETRIES=3
SUPABASE_CONNECTION_RETRY_DELAY=1.0
SUPABASE_CONNECTION_TIMEOUT=30.0
SUPABASE_HEALTH_CHECK_INTERVAL=60.0
```

### Database Schema

The integration expects the following Supabase tables:

```sql
-- Health check table
CREATE TABLE health_check (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Users table (updated for Supabase)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    supabase_user_id UUID UNIQUE,
    profile_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trend analyses table
CREATE TABLE trend_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    analysis_data JSONB,
    keywords TEXT[],
    insights JSONB,
    trend_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Usage Examples

### Basic CRUD Operations

```python
from src.services.supabase_service import SupabaseService

# Initialize service
supabase_service = SupabaseService()

# Create a new user
user_data = {
    "email": "user@example.com",
    "name": "John Doe",
    "supabase_user_id": "uuid-from-supabase-auth"
}
result = await supabase_service.insert("users", user_data, user_id="user123")

# Read users
users = await supabase_service.select("users", {"status": "active"})

# Update user
update_data = {"name": "John Updated"}
await supabase_service.update("users", {"id": user_id}, update_data)

# Delete user
await supabase_service.delete("users", {"id": user_id})
```

### Real-time Subscriptions

```python
from src.services.realtime_service import RealTimeService

# Initialize service
realtime_service = RealTimeService()

# Create subscription
subscription = await realtime_service.create_subscription(
    table_name="trend_analyses",
    event_types=["INSERT", "UPDATE"],
    filter_conditions={"user_id": "user123"}
)

# Set callback for handling events
def handle_event(payload):
    print(f"Real-time event: {payload}")

realtime_service.set_subscription_callback(
    subscription["subscription_id"], 
    handle_event
)

# Unsubscribe when done
await realtime_service.unsubscribe(subscription["subscription_id"])
```

### Migration Management

```python
from src.services.migration_service import MigrationService

# Initialize service
migration_service = MigrationService()

# Start migration
migration = await migration_service.start_migration(
    migration_type=MigrationType.FULL,
    tables=["users", "trend_analyses"],
    dry_run=False
)

# Check status
status = await migration_service.get_migration_status(migration["migration_id"])
print(f"Migration status: {status['status']}")

# Rollback if needed
if status["status"] == "failed":
    await migration_service.rollback_migration(migration["migration_id"])
```

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py --all

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --contract

# Run with coverage
python run_tests.py --all --coverage

# Run in parallel
python run_tests.py --all --parallel 4
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions
- **Contract Tests** (`tests/contract/`): Test API contracts and schemas

### Test Coverage

The test suite provides comprehensive coverage for:
- Supabase client operations
- Database service methods
- Real-time functionality
- Error handling
- Migration processes
- WebSocket connections

## Error Handling

### Common Error Types

1. **Connection Errors**: Network issues, timeouts
2. **Authentication Errors**: Invalid tokens, expired sessions
3. **Permission Errors**: RLS policy violations
4. **Data Validation Errors**: Invalid data format
5. **Rate Limiting**: API rate limits exceeded

### Error Response Format

```json
{
    "detail": "Operation failed: Row level security policy violation (Code: 42501)",
    "status_code": 403,
    "error_type": "APIResponseException",
    "context": "database operation"
}
```

## Monitoring and Logging

### Operation Tracking

All database operations are automatically tracked with:
- Operation ID and type
- Table name and query data
- Execution time and status
- User ID and timestamp
- Error messages (if any)

### Health Monitoring

The system provides health checks for:
- Database connectivity
- Real-time service status
- Migration progress
- WebSocket connections

### Logging

Structured logging is used throughout with:
- Operation context
- Performance metrics
- Error details
- User activity

## Performance Considerations

### Connection Pooling

- Automatic connection management
- Configurable retry logic
- Health check intervals
- Connection timeouts

### Real-time Optimization

- Efficient WebSocket handling
- Event filtering and routing
- Subscription management
- Connection cleanup

### Migration Performance

- Batch processing
- Progress tracking
- Rollback capabilities
- Data validation

## Security

### Row Level Security (RLS)

Supabase RLS policies should be configured for:
- User data isolation
- Admin access controls
- Public data access
- API key restrictions

### Authentication

- JWT token validation
- Session management
- User context tracking
- Permission checks

### Data Protection

- Input validation
- SQL injection prevention
- XSS protection
- CSRF protection

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check Supabase URL and keys
   - Verify network connectivity
   - Check firewall settings

2. **Authentication Errors**
   - Verify JWT token validity
   - Check user permissions
   - Validate RLS policies

3. **Real-time Issues**
   - Check WebSocket connectivity
   - Verify subscription setup
   - Monitor event processing

4. **Migration Problems**
   - Check data compatibility
   - Verify table schemas
   - Monitor migration progress

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check Endpoints

Use health check endpoints to diagnose issues:
- `/api/v1/health/database` - Database connectivity
- `/api/v1/database/operations` - Operation history
- `/api/v1/database/migrate/{id}` - Migration status

## Migration from PostgreSQL

### Migration Process

1. **Backup Data**: Create full backup of existing PostgreSQL database
2. **Setup Supabase**: Create Supabase project and configure tables
3. **Run Migration**: Execute migration scripts to transfer data
4. **Validate Data**: Verify data integrity and completeness
5. **Update Configuration**: Switch to Supabase endpoints
6. **Test Functionality**: Verify all features work correctly

### Migration Scripts

- `src/scripts/migrate_to_supabase.py` - Main migration script
- `src/scripts/validate_migration.py` - Data validation
- `src/scripts/monitor_migration.py` - Progress monitoring

### Rollback Plan

- Keep original PostgreSQL database running
- Maintain data synchronization
- Implement feature flags
- Prepare rollback procedures

## Best Practices

### Development

1. **Use Services**: Always use service classes, not direct Supabase client
2. **Handle Errors**: Implement proper error handling and logging
3. **Test Thoroughly**: Write comprehensive tests for all functionality
4. **Monitor Performance**: Track operation metrics and response times

### Production

1. **Configure RLS**: Set up proper Row Level Security policies
2. **Monitor Health**: Use health check endpoints for monitoring
3. **Handle Failures**: Implement retry logic and fallback mechanisms
4. **Scale Appropriately**: Monitor usage and scale as needed

### Maintenance

1. **Regular Backups**: Maintain data backups and recovery procedures
2. **Update Dependencies**: Keep Supabase SDK and dependencies updated
3. **Monitor Logs**: Review logs for errors and performance issues
4. **Test Migrations**: Test migration scripts in staging environment

## Support

For issues and questions:
1. Check this documentation
2. Review test cases for examples
3. Check logs for error details
4. Consult Supabase documentation
5. Contact development team

## Changelog

### Version 1.0.0
- Initial Supabase integration
- Core CRUD operations
- Real-time features
- Migration tools
- Comprehensive testing
- Documentation

