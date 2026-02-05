# Email Verification API Documentation

This document describes the email verification functionality in the Trend Analysis Platform API.

## Overview

The email verification feature ensures that users verify their email addresses before they can fully use the platform. It consists of:

1. **Email Verification** - Verifies user's email address using a token from verification email
2. **Verification Email Sending** - Automatically sent during user registration

## Endpoints

### 1. Verify Email

**POST** `/api/v1/auth/verify-email`

Verifies a user's email address using a token from the verification email.

#### Request Body

```json
{
  "token": "verification-token-from-email"
}
```

#### Request Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| token | string | Yes | Verification token from email |

#### Response

**Success (200 OK)**
```json
{
  "message": "Email verified successfully. Welcome to the platform!"
}
```

**Error (400 Bad Request)**
```json
{
  "error": "INVALID_TOKEN",
  "message": "Invalid or expired verification token"
}
```

**Error (403 Forbidden)**
```json
{
  "error": "FEATURE_DISABLED",
  "message": "Email verification is currently disabled"
}
```

#### Example Usage

```bash
curl -X POST "http://localhost:8000/api/v1/auth/verify-email" \
  -H "Content-Type: application/json" \
  -d '{"token": "abc123def456"}'
```

## Security Features

### 1. Token Security
- Verification tokens are cryptographically secure
- Tokens expire after 24 hours by default
- Tokens are single-use only
- Tokens are invalidated after successful verification

### 2. Rate Limiting
- Email verification requests are rate limited
- Prevents abuse and token enumeration
- Configurable limits per IP address

### 3. Email Security
- Secure token generation
- Email validation before sending
- Generic error messages to prevent enumeration

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_EMAIL_VERIFICATION` | Enable/disable email verification | `true` |
| `EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS` | Token expiration time | `24` |
| `SENDGRID_API_KEY` | SendGrid API key for emails | Required |
| `FROM_EMAIL` | Sender email address | Required |
| `FRONTEND_URL` | Frontend URL for verification links | Required |

### Feature Flags

```python
# Enable/disable email verification
ENABLE_EMAIL_VERIFICATION = True

# Token expiration time (hours)
EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS = 24

# Email configuration
SENDGRID_API_KEY = "your-sendgrid-key"
FROM_EMAIL = "noreply@trendanalysis.com"
FRONTEND_URL = "https://trendanalysis.com"
```

## User Registration Flow

### 1. User Registration
When a user registers, the system:
1. Creates user account with `is_verified = false`
2. Generates verification token
3. Sends verification email
4. Returns success message

### 2. Email Verification
When user clicks verification link:
1. User submits token via API
2. System validates token
3. System marks user as verified
4. System sends welcome email
5. User can now fully use the platform

## Database Schema

### User Model Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_verified` | boolean | Whether email is verified |
| `email_verification_token` | string | Verification token |
| `email_verification_expires` | datetime | Token expiration time |

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid token |
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
// Verify email
async function verifyEmail(token) {
  const response = await fetch('/api/v1/auth/verify-email', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ token }),
  });
  
  const data = await response.json();
  return data;
}

// Check verification status
async function checkVerificationStatus() {
  const response = await fetch('/api/v1/auth/me', {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });
  
  const user = await response.json();
  return user.is_verified;
}
```

### Python Integration

```python
import requests

# Verify email
def verify_email(token):
    response = requests.post(
        'http://localhost:8000/api/v1/auth/verify-email',
        json={'token': token}
    )
    return response.json()

# Check verification status
def check_verification_status(access_token):
    response = requests.get(
        'http://localhost:8000/api/v1/auth/me',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    user = response.json()
    return user['is_verified']
```

## Testing

### Test Scenarios

1. **Valid Token**
   - Use valid verification token
   - Verify success response
   - Check user is marked as verified

2. **Invalid Token**
   - Use invalid/expired token
   - Verify error response

3. **Already Verified**
   - Use token for already verified user
   - Verify appropriate response

4. **Expired Token**
   - Use expired token
   - Verify error response

### Test Data

```json
{
  "valid_token": "abc123def456",
  "invalid_token": "invalid-token",
  "expired_token": "expired-token"
}
```

## Monitoring and Logging

### Log Events

- Email verification requests
- Token generation
- Email sending
- Successful verifications
- Failed attempts
- Security violations

### Metrics

- Verification requests per minute
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
   - Request new verification email
   - Check token expiration time

3. **Feature disabled**
   - Check `ENABLE_EMAIL_VERIFICATION` setting
   - Verify configuration

4. **Invalid token**
   - Check token format
   - Verify token hasn't been used

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
LOG_LEVEL = "DEBUG"
ENABLE_REQUEST_LOGGING = True
```

## Security Considerations

1. **Never log sensitive data** (tokens)
2. **Use HTTPS** in production
3. **Implement rate limiting**
4. **Monitor for abuse**
5. **Regular security audits**
6. **Keep dependencies updated**

## Best Practices

1. **Clear user messaging** about verification status
2. **Resend verification** option for users
3. **Graceful handling** of verification failures
4. **User-friendly error messages**
5. **Proper email templates**
6. **Mobile-responsive** verification pages

## Support

For issues and questions:
- Check API documentation
- Review error messages
- Check server logs
- Contact support team
