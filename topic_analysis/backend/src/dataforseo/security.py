"""
Security utilities for DataForSEO features

Provides security measures including input validation,
rate limiting, authentication, and data protection.
"""

import asyncio
import hashlib
import hmac
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .cache_integration import cache_manager

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event data class"""
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime

class InputValidator:
    """Validates and sanitizes input data"""
    
    def __init__(self):
        self.max_string_length = 1000
        self.max_array_length = 100
        self.allowed_characters = re.compile(r'^[a-zA-Z0-9\s\-_.,!?@#$%^&*()+=\[\]{}|;:"<>/\\]+$')
        self.sql_injection_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
            r'(\b(OR|AND)\s+\w+\s*LIKE\s+[\'"]%)',
            r'(\b(OR|AND)\s+\w+\s*IN\s*\([^)]+\))',
            r'(\b(OR|AND)\s+\w+\s*BETWEEN\s+\d+\s+AND\s+\d+)',
            r'(\b(OR|AND)\s+\w+\s*IS\s+NULL)',
            r'(\b(OR|AND)\s+\w+\s*IS\s+NOT\s+NULL)',
            r'(\b(OR|AND)\s+\w+\s*EXISTS\s*\([^)]+\))',
            r'(\b(OR|AND)\s+\w+\s*NOT\s+EXISTS\s*\([^)]+\))'
        ]
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<link[^>]*>.*?</link>',
            r'<meta[^>]*>.*?</meta>',
            r'<style[^>]*>.*?</style>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'onfocus\s*=',
            r'onblur\s*=',
            r'onchange\s*=',
            r'onsubmit\s*=',
            r'onreset\s*=',
            r'onselect\s*=',
            r'onkeydown\s*=',
            r'onkeyup\s*=',
            r'onkeypress\s*=',
            r'onmousedown\s*=',
            r'onmouseup\s*=',
            r'onmousemove\s*=',
            r'onmouseout\s*=',
            r'onmouseenter\s*=',
            r'onmouseleave\s*=',
            r'ondblclick\s*=',
            r'oncontextmenu\s*=',
            r'ondrag\s*=',
            r'ondragstart\s*=',
            r'ondragend\s*=',
            r'ondragenter\s*=',
            r'ondragleave\s*=',
            r'ondragover\s*=',
            r'ondrop\s*=',
            r'onscroll\s*=',
            r'onresize\s*=',
            r'onabort\s*=',
            r'oncanplay\s*=',
            r'oncanplaythrough\s*=',
            r'ondurationchange\s*=',
            r'onemptied\s*=',
            r'onended\s*=',
            r'onerror\s*=',
            r'onloadeddata\s*=',
            r'onloadedmetadata\s*=',
            r'onloadstart\s*=',
            r'onpause\s*=',
            r'onplay\s*=',
            r'onplaying\s*=',
            r'onprogress\s*=',
            r'onratechange\s*=',
            r'onseeked\s*=',
            r'onseeking\s*=',
            r'onstalled\s*=',
            r'onsuspend\s*=',
            r'ontimeupdate\s*=',
            r'onvolumechange\s*=',
            r'onwaiting\s*='
        ]
    
    def validate_string(self, value: str, field_name: str) -> Tuple[bool, str]:
        """Validate string input"""
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        
        if len(value) > self.max_string_length:
            return False, f"{field_name} exceeds maximum length of {self.max_string_length}"
        
        if not value.strip():
            return False, f"{field_name} cannot be empty"
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, f"{field_name} contains potentially malicious SQL content"
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, f"{field_name} contains potentially malicious script content"
        
        # Check for allowed characters
        if not self.allowed_characters.match(value):
            return False, f"{field_name} contains invalid characters"
        
        return True, "Valid"
    
    def validate_array(self, value: List[str], field_name: str) -> Tuple[bool, str]:
        """Validate array input"""
        if not isinstance(value, list):
            return False, f"{field_name} must be an array"
        
        if len(value) > self.max_array_length:
            return False, f"{field_name} exceeds maximum length of {self.max_array_length}"
        
        for i, item in enumerate(value):
            if not isinstance(item, str):
                return False, f"{field_name}[{i}] must be a string"
            
            is_valid, error = self.validate_string(item, f"{field_name}[{i}]")
            if not is_valid:
                return False, error
        
        return True, "Valid"
    
    def validate_number(self, value: Any, field_name: str, min_val: float = None, max_val: float = None) -> Tuple[bool, str]:
        """Validate number input"""
        if not isinstance(value, (int, float)):
            return False, f"{field_name} must be a number"
        
        if min_val is not None and value < min_val:
            return False, f"{field_name} must be at least {min_val}"
        
        if max_val is not None and value > max_val:
            return False, f"{field_name} must be at most {max_val}"
        
        return True, "Valid"
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Strip whitespace
        value = value.strip()
        
        # Limit length
        value = value[:self.max_string_length]
        
        return value
    
    def validate_trend_request(self, request: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate trend analysis request"""
        errors = []
        
        # Validate subtopics
        if 'subtopics' in request:
            is_valid, error = self.validate_array(request['subtopics'], 'subtopics')
            if not is_valid:
                errors.append(error)
        
        # Validate location
        if 'location' in request:
            is_valid, error = self.validate_string(request['location'], 'location')
            if not is_valid:
                errors.append(error)
        
        # Validate time_range
        if 'time_range' in request:
            valid_ranges = ['1m', '3m', '6m', '12m', '24m']
            if request['time_range'] not in valid_ranges:
                errors.append(f"time_range must be one of: {', '.join(valid_ranges)}")
        
        return len(errors) == 0, errors
    
    def validate_keyword_request(self, request: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate keyword research request"""
        errors = []
        
        # Validate seed_keywords
        if 'seed_keywords' in request:
            is_valid, error = self.validate_array(request['seed_keywords'], 'seed_keywords')
            if not is_valid:
                errors.append(error)
        
        # Validate max_difficulty
        if 'max_difficulty' in request:
            is_valid, error = self.validate_number(request['max_difficulty'], 'max_difficulty', 0, 100)
            if not is_valid:
                errors.append(error)
        
        # Validate min_volume
        if 'min_volume' in request:
            is_valid, error = self.validate_number(request['min_volume'], 'min_volume', 0)
            if not is_valid:
                errors.append(error)
        
        # Validate intent_types
        if 'intent_types' in request:
            valid_intents = ['INFORMATIONAL', 'COMMERCIAL', 'TRANSACTIONAL']
            for intent in request['intent_types']:
                if intent not in valid_intents:
                    errors.append(f"intent_types must be one of: {', '.join(valid_intents)}")
                    break
        
        # Validate max_results
        if 'max_results' in request:
            is_valid, error = self.validate_number(request['max_results'], 'max_results', 1, 1000)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors

class RateLimiter:
    """Implements rate limiting for API requests"""
    
    def __init__(self):
        self.rate_limits = {
            'trend_analysis': {'requests_per_minute': 60, 'requests_per_hour': 1000},
            'keyword_research': {'requests_per_minute': 120, 'requests_per_hour': 2000},
            'suggestions': {'requests_per_minute': 30, 'requests_per_hour': 500}
        }
        self.request_counts = {}
    
    async def check_rate_limit(self, user_id: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        now = time.time()
        minute_key = f"{user_id}:{endpoint}:minute:{int(now // 60)}"
        hour_key = f"{user_id}:{endpoint}:hour:{int(now // 3600)}"
        
        # Get current counts
        minute_count = await cache_manager.get(minute_key) or 0
        hour_count = await cache_manager.get(hour_key) or 0
        
        # Check limits
        limits = self.rate_limits.get(endpoint, {'requests_per_minute': 60, 'requests_per_hour': 1000})
        
        if minute_count >= limits['requests_per_minute']:
            return False, {
                'error': 'Rate limit exceeded',
                'limit_type': 'minute',
                'limit': limits['requests_per_minute'],
                'current': minute_count,
                'reset_time': int((now // 60 + 1) * 60)
            }
        
        if hour_count >= limits['requests_per_hour']:
            return False, {
                'error': 'Rate limit exceeded',
                'limit_type': 'hour',
                'limit': limits['requests_per_hour'],
                'current': hour_count,
                'reset_time': int((now // 3600 + 1) * 3600)
            }
        
        # Increment counters
        await cache_manager.set(minute_key, minute_count + 1, 60)  # 1 minute TTL
        await cache_manager.set(hour_key, hour_count + 1, 3600)   # 1 hour TTL
        
        return True, {
            'minute_count': minute_count + 1,
            'hour_count': hour_count + 1,
            'minute_limit': limits['requests_per_minute'],
            'hour_limit': limits['requests_per_hour']
        }
    
    async def get_rate_limit_status(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = time.time()
        minute_key = f"{user_id}:{endpoint}:minute:{int(now // 60)}"
        hour_key = f"{user_id}:{endpoint}:hour:{int(now // 3600)}"
        
        minute_count = await cache_manager.get(minute_key) or 0
        hour_count = await cache_manager.get(hour_key) or 0
        
        limits = self.rate_limits.get(endpoint, {'requests_per_minute': 60, 'requests_per_hour': 1000})
        
        return {
            'endpoint': endpoint,
            'minute_count': minute_count,
            'hour_count': hour_count,
            'minute_limit': limits['requests_per_minute'],
            'hour_limit': limits['requests_per_hour'],
            'minute_remaining': max(0, limits['requests_per_minute'] - minute_count),
            'hour_remaining': max(0, limits['requests_per_hour'] - hour_count)
        }

class SecurityAuditor:
    """Audits security events and monitors for threats"""
    
    def __init__(self):
        self.security_events = []
        self.max_events = 10000
        self.threat_patterns = {
            'sql_injection': r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)',
            'xss': r'<script[^>]*>.*?</script>',
            'path_traversal': r'\.\./|\.\.\\|\.\.%2f|\.\.%5c',
            'command_injection': r'[;&|`$]',
            'ldap_injection': r'[()=*!|&]',
            'nosql_injection': r'(\$where|\$ne|\$gt|\$lt|\$regex)'
        }
    
    def log_security_event(self, event_type: str, severity: str, user_id: str = None, 
                          ip_address: str = None, details: Dict[str, Any] = None):
        """Log a security event"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            timestamp=datetime.utcnow()
        )
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        # Log to file
        logger.warning(f"Security event: {event_type} - {severity} - {user_id} - {ip_address}")
        
        # Alert on high severity events
        if severity == 'HIGH':
            self._alert_high_severity(event)
    
    def _alert_high_severity(self, event: SecurityEvent):
        """Alert on high severity security events"""
        # In production, this would send alerts to security team
        logger.critical(f"HIGH SEVERITY SECURITY EVENT: {event.event_type} - {event.details}")
    
    def detect_threats(self, input_data: str, user_id: str = None, ip_address: str = None) -> List[str]:
        """Detect potential security threats in input data"""
        threats = []
        
        for threat_type, pattern in self.threat_patterns.items():
            if re.search(pattern, input_data, re.IGNORECASE):
                threats.append(threat_type)
                self.log_security_event(
                    event_type=f"threat_detected_{threat_type}",
                    severity="HIGH",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={
                        'threat_type': threat_type,
                        'input_data': input_data[:100],  # Truncate for logging
                        'pattern': pattern
                    }
                )
        
        return threats
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events 
            if event.timestamp >= cutoff_time
        ]
        
        event_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        return {
            'total_events': len(recent_events),
            'event_types': event_counts,
            'severity_counts': severity_counts,
            'time_range_hours': hours,
            'high_severity_events': [
                event for event in recent_events if event.severity == 'HIGH'
            ]
        }

class DataProtector:
    """Protects sensitive data and implements data privacy measures"""
    
    def __init__(self):
        self.sensitive_fields = ['api_key', 'password', 'token', 'secret', 'credential']
        self.mask_char = '*'
        self.mask_length = 8
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionaries"""
        masked_data = {}
        
        for key, value in data.items():
            if any(field in key.lower() for field in self.sensitive_fields):
                if isinstance(value, str):
                    masked_data[key] = self.mask_string(value)
                else:
                    masked_data[key] = f"<{type(value).__name__}>"
            elif isinstance(value, dict):
                masked_data[key] = self.mask_sensitive_data(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data
    
    def mask_string(self, value: str) -> str:
        """Mask string value"""
        if len(value) <= 4:
            return self.mask_char * len(value)
        
        if len(value) <= self.mask_length:
            return value[0] + self.mask_char * (len(value) - 2) + value[-1]
        
        return value[:2] + self.mask_char * (self.mask_length - 4) + value[-2:]
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_data_integrity(self, data: str, hash_value: str) -> bool:
        """Verify data integrity using hash"""
        return hmac.compare_digest(self.hash_sensitive_data(data), hash_value)

# Global instances
input_validator = InputValidator()
rate_limiter = RateLimiter()
security_auditor = SecurityAuditor()
data_protector = DataProtector()
