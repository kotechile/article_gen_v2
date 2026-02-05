"""
Password hashing service using bcrypt.
"""
import bcrypt
from typing import Tuple
import re

class PasswordService:
    """Service for password hashing and verification."""
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength requirements."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be no more than 128 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"
        
        # Check for common weak patterns
        if self._is_common_password(password):
            return False, "Password is too common and easily guessable"
        
        # Check for repeated characters (only if 4+ consecutive)
        if self._has_repeated_characters(password):
            return False, "Password contains too many repeated characters"
        
        # Check for obvious sequential patterns (only long sequences)
        if self._has_obvious_sequential_patterns(password):
            return False, "Password contains obvious sequential patterns that are easy to guess"
        
        return True, "Password is valid"
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list."""
        common_passwords = {
            "password", "123456", "123456789", "qwerty", "abc123", "password123",
            "admin", "letmein", "welcome", "monkey", "1234567890", "dragon",
            "master", "hello", "freedom", "whatever", "qazwsx", "trustno1",
            "jordan23", "harley", "ranger", "jennifer", "hunter", "fuck",
            "jordan", "love", "asshole", "2000", "robert", "andrew", "password1",
            "superman", "qwertyuiop", "123123", "dallas", "mustang", "access",
            "shadow", "michael", "jordan", "master", "jennifer", "joshua",
            "monkey", "abcd1234", "qwerty", "password", "1234567890", "welcome"
        }
        return password.lower() in common_passwords
    
    def _has_repeated_characters(self, password: str) -> bool:
        """Check if password has too many repeated characters."""
        # Check for 4 or more consecutive identical characters
        for i in range(len(password) - 3):
            if password[i] == password[i + 1] == password[i + 2] == password[i + 3]:
                return True
        return False
    
    def _has_obvious_sequential_patterns(self, password: str) -> bool:
        """Check if password has obvious sequential patterns."""
        # Check for long numeric sequences (4+ consecutive digits)
        for i in range(len(password) - 3):
            if (password[i].isdigit() and password[i + 1].isdigit() and 
                password[i + 2].isdigit() and password[i + 3].isdigit()):
                if (int(password[i + 1]) == int(password[i]) + 1 and 
                    int(password[i + 2]) == int(password[i]) + 2 and
                    int(password[i + 3]) == int(password[i]) + 3):
                    return True
        
        # Check for long alphabetic sequences (4+ consecutive letters)
        for i in range(len(password) - 3):
            if (password[i].isalpha() and password[i + 1].isalpha() and 
                password[i + 2].isalpha() and password[i + 3].isalpha()):
                if (ord(password[i + 1].lower()) == ord(password[i].lower()) + 1 and 
                    ord(password[i + 2].lower()) == ord(password[i].lower()) + 2 and
                    ord(password[i + 3].lower()) == ord(password[i].lower()) + 3):
                    return True
        
        # Check for obvious keyboard patterns
        obvious_patterns = [
            "qwerty", "asdfgh", "zxcvbn", "123456789", "987654321", 
            "abcdefgh", "hgfedcba", "qwertyuiop", "asdfghjkl"
        ]
        
        password_lower = password.lower()
        for pattern in obvious_patterns:
            if pattern in password_lower:
                return True
        
        return False
    
    def get_password_strength_score(self, password: str) -> int:
        """Get password strength score (0-100)."""
        score = 0
        
        # Length score (0-30 points)
        if len(password) >= 8:
            score += 10
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10
        
        # Character variety score (0-40 points)
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 10
        
        # Complexity score (0-30 points)
        if len(password) >= 8 and not self._is_common_password(password):
            score += 10
        if not self._has_repeated_characters(password):
            score += 10
        if not self._has_obvious_sequential_patterns(password):
            score += 10
        
        return min(score, 100)
    
    def get_password_strength_level(self, password: str) -> str:
        """Get password strength level."""
        score = self.get_password_strength_score(password)
        
        if score < 30:
            return "Very Weak"
        elif score < 50:
            return "Weak"
        elif score < 70:
            return "Fair"
        elif score < 90:
            return "Good"
        else:
            return "Strong"
    
    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if password needs to be rehashed with current settings."""
        try:
            # Extract rounds from bcrypt hash
            rounds = int(hashed_password.split('$')[2])
            return rounds < self.rounds
        except (ValueError, IndexError):
            return True

# Create singleton instance
password_service = PasswordService()
