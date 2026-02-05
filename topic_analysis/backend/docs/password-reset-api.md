# Password Reset API Documentation

This document describes the password reset functionality in the Trend Analysis Platform API.

## Overview

The password reset feature allows users to reset their passwords when they forget them. It consists of two main endpoints:

1. **Request Password Reset** - Initiates the password reset process
2. **Confirm Password Reset** - Completes the password reset with a new password

## Endpoints

### 1. Request Password Reset

**POST** `/api/v1/auth/request-password-reset`

Initiates a password reset process by sending a reset email to the user.

#### Request Body

```json
{
  "email": "user@example.com"
}
```

#### Request Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string (email) | Yes | User's email address |

#### Response

**Success (200 OK)**
```json
{
  "message": "If an account with that email exists, a password reset link has been sent."
}
```

**Error (400 Bad Request)**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "field": "email",
      "reason": "Invalid email format"
    }
  ]
}
```

**Error (403 Forbidden)**
```json
{
  "error": "FEATURE_DISABLED",
  "message": "Password reset is currently disabled"
}
```

#### Example Usage

```bash
curl -X POST "http://localhost:8000/api/v1/auth/request-password-reset" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com"}'
```

### 2. Confirm Password Reset

**POST** `/api/v1/auth/confirm-password-reset`

Completes the password reset process using a token from the reset email.

#### Request Body

```json
{
  "token": "reset-token-from-email",
  "new_password": "NewSecurePassword123!"
}
```

#### Request Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| token | string | Yes | Reset token from email |
| new_password | string | Yes | New password (8+ chars, uppercase, lowercase, digits, special chars) |

#### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)

#### Response

**Success (200 OK)**
```json
{
  "message": "Password has been reset successfully."
}
```

**Error (400 Bad Request)**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "field": "new_password",
      "reason": "Password must contain at least one uppercase letter"
    }
  ]
}
```

**Error (400 Bad Request)**
```json
{
  "error": "INVALID_TOKEN",
  "message": "Invalid or expired password reset token"
}
```

**Error (403 Forbidden)**
```json
{
  "error": "FEATURE_DISABLED",
  "message": "Password reset is currently disabled"
}
```

#### Example Usage

```bash
curl -X POST "http://localhost:8000/api/v1/auth/confirm-password-reset" \
  -H "Content-Type: application/json" \
  -d '{
    "token": "abc123def456",
    "new_password": "NewSecurePassword123!"
  }'
```

## Security Features

### 1. Token Security
- Reset tokens are cryptographically secure
- Tokens expire after 1 hour by default
- Tokens are single-use only
- Tokens are invalidated after successful reset

### 2. Rate Limiting
- Password reset requests are rate limited
- Prevents abuse and email bombing
- Configurable limits per IP address

### 3. Email Security
- Generic response message to prevent email enumeration
- Secure token generation
- Email validation before sending

### 4. Password Security
- Strong password requirements
- Password strength validation
- Secure password hashing with bcrypt

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_PASSWORD_RESET` | Enable/disable password reset | `true` |
| `PASSWORD_RESET_TOKEN_EXPIRE_HOURS` | Token expiration time | `1` |
| `SENDGRID_API_KEY` | SendGrid API key for emails | Required |
| `FROM_EMAIL` | Sender email address | Required |
| `FRONTEND_URL` | Frontend URL for reset links | Required |

### Feature Flags

```python
# Enable/disable password reset
ENABLE_PASSWORD_RESET = True

# Token expiration time (hours)
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 1

# Email configuration
SENDGRID_API_KEY = "your-sendgrid-key"
FROM_EMAIL = "noreply@trendanalysis.com"
FRONTEND_URL = "https://trendanalysis.com"
```

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 403 | Forbidden - Feature disabled |
| 422 | Unprocessable Entity - Validation error |
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

## Integration Examples

### Frontend Integration

```javascript
// Request password reset
async function requestPasswordReset(email) {
  const response = await fetch('/api/v1/auth/request-password-reset', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email }),
  });
  
  const data = await response.json();
  return data;
}

// Confirm password reset
async function confirmPasswordReset(token, newPassword) {
  const response = await fetch('/api/v1/auth/confirm-password-reset', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  
  const data = await response.json();
  return data;
}
```

### Python Integration

```python
import requests

# Request password reset
def request_password_reset(email):
    response = requests.post(
        'http://localhost:8000/api/v1/auth/request-password-reset',
        json={'email': email}
    )
    return response.json()

# Confirm password reset
def confirm_password_reset(token, new_password):
    response = requests.post(
        'http://localhost:8000/api/v1/auth/confirm-password-reset',
        json={'token': token, 'new_password': new_password}
    )
    return response.json()
```

## Testing

### Test Scenarios

1. **Valid Request**
   - Send valid email address
   - Verify success response
   - Check email is sent

2. **Invalid Email**
   - Send invalid email format
   - Verify validation error

3. **Non-existent User**
   - Send email for non-existent user
   - Verify generic success response (security)

4. **Valid Token Reset**
   - Use valid reset token
   - Set new password
   - Verify success

5. **Invalid Token**
   - Use expired/invalid token
   - Verify error response

6. **Weak Password**
   - Use weak password
   - Verify validation error

### Test Data

```json
{
  "valid_email": "test@example.com",
  "invalid_email": "invalid-email",
  "valid_token": "abc123def456",
  "invalid_token": "expired-token",
  "strong_password": "StrongPassword123!",
  "weak_password": "weak"
}
```

## Monitoring and Logging

### Log Events

- Password reset requests
- Token generation
- Email sending
- Successful resets
- Failed attempts
- Security violations

### Metrics

- Reset requests per minute
- Success/failure rates
- Token expiration rates
- Email delivery rates

## Troubleshooting

### Common Issues

1. **Email not received**
   - Check SendGrid configuration
   - Verify email address
   - Check spam folder

2. **Token expired**
   - Request new reset
   - Check token expiration time

3. **Password validation failed**
   - Check password requirements
   - Ensure all criteria are met

4. **Feature disabled**
   - Check `ENABLE_PASSWORD_RESET` setting
   - Verify configuration

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
LOG_LEVEL = "DEBUG"
ENABLE_REQUEST_LOGGING = True
```

## Security Considerations

1. **Never log sensitive data** (tokens, passwords)
2. **Use HTTPS** in production
3. **Implement rate limiting**
4. **Monitor for abuse**
5. **Regular security audits**
6. **Keep dependencies updated**

## Support

For issues and questions:
- Check API documentation
- Review error messages
- Check server logs
- Contact support team
