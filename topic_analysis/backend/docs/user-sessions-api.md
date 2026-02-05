# User Sessions API Documentation

This document describes the user sessions management functionality in the Trend Analysis Platform API.

## Overview

The user sessions feature allows users to manage their active sessions across different devices and browsers. It provides:

1. **List Active Sessions** - View all active sessions for the current user
2. **Revoke Specific Session** - Revoke a specific session by ID
3. **Revoke All Sessions** - Revoke all sessions for the current user

## Endpoints

### 1. Get User Sessions

**GET** `/api/v1/users/sessions`

Retrieves all active sessions for the current user.

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token for authentication |

#### Response

**Success (200 OK)**
```json
{
  "sessions": [
    {
      "id": "session-uuid",
      "user_id": "user-uuid",
      "device_info": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "ip_address": "192.168.1.100"
      },
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "is_active": true,
      "expires_at": "2024-01-02T00:00:00Z",
      "created_at": "2024-01-01T00:00:00Z",
      "last_accessed": "2024-01-01T12:00:00Z"
    }
  ]
}
```

**Error (401 Unauthorized)**
```json
{
  "error": "UNAUTHORIZED",
  "message": "Invalid authentication credentials"
}
```

#### Example Usage

```bash
curl -X GET "http://localhost:8000/api/v1/users/sessions" \
  -H "Authorization: Bearer your-access-token"
```

### 2. Revoke User Session

**DELETE** `/api/v1/users/sessions/{session_id}`

Revokes a specific session by its ID.

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token for authentication |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string (UUID) | Yes | ID of the session to revoke |

#### Response

**Success (200 OK)**
```json
{
  "message": "Session revoked successfully"
}
```

**Error (404 Not Found)**
```json
{
  "error": "NOT_FOUND",
  "message": "Session not found"
}
```

**Error (401 Unauthorized)**
```json
{
  "error": "UNAUTHORIZED",
  "message": "Invalid authentication credentials"
}
```

#### Example Usage

```bash
curl -X DELETE "http://localhost:8000/api/v1/users/sessions/session-uuid" \
  -H "Authorization: Bearer your-access-token"
```

### 3. Revoke All User Sessions

**DELETE** `/api/v1/users/sessions`

Revokes all active sessions for the current user.

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token for authentication |

#### Response

**Success (200 OK)**
```json
{
  "message": "All sessions revoked successfully"
}
```

**Error (401 Unauthorized)**
```json
{
  "error": "UNAUTHORIZED",
  "message": "Invalid authentication credentials"
}
```

#### Example Usage

```bash
curl -X DELETE "http://localhost:8000/api/v1/users/sessions" \
  -H "Authorization: Bearer your-access-token"
```

## Security Features

### 1. Authentication Required
- All endpoints require valid JWT authentication
- Users can only access their own sessions
- Session ownership is validated before operations

### 2. Session Security
- Sessions are identified by unique UUIDs
- JWT tokens are invalidated when sessions are revoked
- Device information is tracked for security monitoring

### 3. Rate Limiting
- Session management operations are rate limited
- Prevents abuse and brute force attacks
- Configurable limits per user

## Session Data Structure

### Session Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique session identifier |
| `user_id` | string (UUID) | ID of the user who owns the session |
| `device_info` | object | Device and browser information |
| `ip_address` | string | IP address of the session |
| `user_agent` | string | User agent string |
| `is_active` | boolean | Whether the session is active |
| `expires_at` | string (ISO 8601) | Session expiration time |
| `created_at` | string (ISO 8601) | Session creation time |
| `last_accessed` | string (ISO 8601) | Last activity time |

### Device Info Object

| Field | Type | Description |
|-------|------|-------------|
| `user_agent` | string | Browser user agent string |
| `ip_address` | string | IP address of the device |

## Use Cases

### 1. Security Monitoring
- Users can monitor their active sessions
- Identify suspicious or unauthorized access
- Revoke sessions from unknown devices

### 2. Device Management
- Manage sessions across multiple devices
- Revoke sessions from lost or stolen devices
- Clean up old or unused sessions

### 3. Account Security
- Revoke all sessions when changing password
- Revoke all sessions when account is compromised
- Monitor session activity for security

## Integration Examples

### Frontend Integration

```javascript
// Get user sessions
async function getUserSessions(accessToken) {
  const response = await fetch('/api/v1/users/sessions', {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });
  
  const data = await response.json();
  return data.sessions;
}

// Revoke specific session
async function revokeSession(sessionId, accessToken) {
  const response = await fetch(`/api/v1/users/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });
  
  const data = await response.json();
  return data;
}

// Revoke all sessions
async function revokeAllSessions(accessToken) {
  const response = await fetch('/api/v1/users/sessions', {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });
  
  const data = await response.json();
  return data;
}
```

### Python Integration

```python
import requests

# Get user sessions
def get_user_sessions(access_token):
    response = requests.get(
        'http://localhost:8000/api/v1/users/sessions',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    return response.json()

# Revoke specific session
def revoke_session(session_id, access_token):
    response = requests.delete(
        f'http://localhost:8000/api/v1/users/sessions/{session_id}',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    return response.json()

# Revoke all sessions
def revoke_all_sessions(access_token):
    response = requests.delete(
        'http://localhost:8000/api/v1/users/sessions',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    return response.json()
```

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 401 | Unauthorized - Invalid or missing authentication |
| 404 | Not Found - Session not found |
| 422 | Unprocessable Entity - Invalid session ID format |
| 500 | Internal Server Error - Server error |

### Error Response Format

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "field": "specific error details"
  }
}
```

## Testing

### Test Scenarios

1. **Get Sessions**
   - Authenticated user can view their sessions
   - Unauthenticated user gets 401 error
   - Empty list returned for user with no sessions

2. **Revoke Session**
   - User can revoke their own session
   - User cannot revoke another user's session
   - Invalid session ID returns 404 error

3. **Revoke All Sessions**
   - User can revoke all their sessions
   - All sessions are marked as inactive
   - User must re-authenticate after revoking all

### Test Data

```json
{
  "valid_session_id": "123e4567-e89b-12d3-a456-426614174000",
  "invalid_session_id": "invalid-uuid",
  "other_user_session_id": "987fcdeb-51a2-43d1-b789-123456789abc"
}
```

## Monitoring and Logging

### Log Events

- Session retrieval requests
- Session revocation requests
- Security violations
- Failed authentication attempts

### Metrics

- Active sessions per user
- Session revocation rates
- Authentication failures
- Security violations

## Troubleshooting

### Common Issues

1. **Session not found**
   - Check session ID format (must be valid UUID)
   - Verify session belongs to current user
   - Check if session was already revoked

2. **Authentication errors**
   - Verify JWT token is valid
   - Check token expiration
   - Ensure proper Authorization header

3. **Permission denied**
   - Verify user owns the session
   - Check user account status
   - Verify session is active

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
LOG_LEVEL = "DEBUG"
ENABLE_REQUEST_LOGGING = True
```

## Security Considerations

1. **Never log sensitive data** (session tokens, JWT)
2. **Use HTTPS** in production
3. **Implement rate limiting**
4. **Monitor for abuse**
5. **Regular security audits**
6. **Keep dependencies updated**

## Best Practices

1. **Regular session cleanup** - Remove old inactive sessions
2. **User education** - Help users understand session management
3. **Security monitoring** - Track unusual session patterns
4. **Graceful handling** - Provide clear error messages
5. **Mobile support** - Ensure mobile-friendly session management
6. **Accessibility** - Make session management accessible

## Support

For issues and questions:
- Check API documentation
- Review error messages
- Check server logs
- Contact support team
