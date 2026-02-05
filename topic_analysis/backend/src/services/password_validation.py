"""
Password strength validation service
"""
import re
import hashlib
import secrets
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
# import zxcvbn  # Optional dependency

class PasswordStrength(Enum):
    """Password strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class PasswordValidationResult:
    """Result of password validation"""
    is_valid: bool
    strength: PasswordStrength
    score: int  # 0-100
    feedback: List[str]
    suggestions: List[str]
    requirements_met: Dict[str, bool]
    entropy: float
    crack_time: str
    crack_time_seconds: float

@dataclass
class PasswordRequirements:
    """Password requirements configuration"""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    min_special_chars: int = 1
    max_consecutive_chars: int = 3
    max_repeated_chars: int = 3
    forbidden_patterns: List[str] = None
    require_no_common_passwords: bool = True
    require_no_user_info: bool = True
    min_entropy: float = 50.0
    min_zxcvbn_score: int = 3

class PasswordValidator:
    """Password strength validator"""
    
    def __init__(self, requirements: Optional[PasswordRequirements] = None):
        self.requirements = requirements or PasswordRequirements()
        self.common_passwords = self._load_common_passwords()
        
    def _load_common_passwords(self) -> set:
        """Load common passwords for checking"""
        # In a real implementation, this would load from a file or database
        return {
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey",
            "1234567890", "dragon", "master", "hello", "freedom",
            "whatever", "qazwsx", "trustno1", "jordan", "harley",
            "ranger", "jennifer", "joshua", "hunter", "buster",
            "soccer", "hockey", "killer", "george", "sexy",
            "andrew", "charlie", "superman", "asshole", "fuckyou",
            "dallas", "jessica", "panties", "pepper", "1234",
            "12345", "1234567", "12345678", "1234567890", "qwertyuiop",
            "asdfghjkl", "zxcvbnm", "1q2w3e4r", "qwerty123", "password1",
            "123456789", "1234567890", "qwertyui", "asdfghjk", "zxcvbnm1"
        }
    
    def validate_password(self, password: str, user_info: Optional[Dict] = None) -> PasswordValidationResult:
        """
        Validate password strength
        
        Args:
            password: Password to validate
            user_info: Optional user information for context checking
            
        Returns:
            PasswordValidationResult with validation details
        """
        if not password:
            return PasswordValidationResult(
                is_valid=False,
                strength=PasswordStrength.VERY_WEAK,
                score=0,
                feedback=["Password is required"],
                suggestions=["Please enter a password"],
                requirements_met={},
                entropy=0.0,
                crack_time="instant",
                crack_time_seconds=0.0
            )
        
        # Basic length check
        if len(password) < self.requirements.min_length:
            return PasswordValidationResult(
                is_valid=False,
                strength=PasswordStrength.VERY_WEAK,
                score=0,
                feedback=[f"Password must be at least {self.requirements.min_length} characters long"],
                suggestions=[f"Use at least {self.requirements.min_length} characters"],
                requirements_met={},
                entropy=0.0,
                crack_time="instant",
                crack_time_seconds=0.0
            )
        
        if len(password) > self.requirements.max_length:
            return PasswordValidationResult(
                is_valid=False,
                strength=PasswordStrength.VERY_WEAK,
                score=0,
                feedback=[f"Password must be no more than {self.requirements.max_length} characters long"],
                suggestions=[f"Use no more than {self.requirements.max_length} characters"],
                requirements_met={},
                entropy=0.0,
                crack_time="instant",
                crack_time_seconds=0.0
            )
        
        # Check requirements
        requirements_met = self._check_requirements(password)
        
        # Use zxcvbn for advanced analysis (if available)
        try:
            import zxcvbn
            zxcvbn_result = zxcvbn.zxcvbn(password, user_inputs=user_info or [])
        except ImportError:
            # Fallback to basic analysis
            zxcvbn_result = {
                'score': 0,
                'feedback': {'warning': '', 'suggestions': []},
                'crack_times_display': {'offline_slow_hashing_1e4_per_second': 'instant'},
                'crack_times_seconds': {'offline_slow_hashing_1e4_per_second': 0}
            }
        
        # Calculate entropy
        entropy = self._calculate_entropy(password)
        
        # Determine strength and score
        strength, score = self._determine_strength(zxcvbn_result, entropy, requirements_met)
        
        # Generate feedback and suggestions
        feedback, suggestions = self._generate_feedback(password, requirements_met, zxcvbn_result)
        
        # Check if password meets minimum requirements
        is_valid = (
            all(requirements_met.values()) and
            zxcvbn_result['score'] >= self.requirements.min_zxcvbn_score and
            entropy >= self.requirements.min_entropy
        )
        
        return PasswordValidationResult(
            is_valid=is_valid,
            strength=strength,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            requirements_met=requirements_met,
            entropy=entropy,
            crack_time=zxcvbn_result['crack_times_display']['offline_slow_hashing_1e4_per_second'],
            crack_time_seconds=zxcvbn_result['crack_times_seconds']['offline_slow_hashing_1e4_per_second']
        )
    
    def _check_requirements(self, password: str) -> Dict[str, bool]:
        """Check if password meets all requirements"""
        requirements = {}
        
        # Length requirements
        requirements['min_length'] = len(password) >= self.requirements.min_length
        requirements['max_length'] = len(password) <= self.requirements.max_length
        
        # Character type requirements
        requirements['uppercase'] = bool(re.search(r'[A-Z]', password)) if self.requirements.require_uppercase else True
        requirements['lowercase'] = bool(re.search(r'[a-z]', password)) if self.requirements.require_lowercase else True
        requirements['digits'] = bool(re.search(r'\d', password)) if self.requirements.require_digits else True
        requirements['special_chars'] = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)) if self.requirements.require_special_chars else True
        
        # Special character count
        special_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', password))
        requirements['min_special_chars'] = special_count >= self.requirements.min_special_chars
        
        # Consecutive character check
        requirements['max_consecutive'] = self._check_consecutive_chars(password)
        
        # Repeated character check
        requirements['max_repeated'] = self._check_repeated_chars(password)
        
        # Pattern checks
        requirements['no_forbidden_patterns'] = self._check_forbidden_patterns(password)
        requirements['no_common_passwords'] = password.lower() not in self.common_passwords if self.requirements.require_no_common_passwords else True
        
        return requirements
    
    def _check_consecutive_chars(self, password: str) -> bool:
        """Check for consecutive characters"""
        if self.requirements.max_consecutive_chars <= 0:
            return True
        
        consecutive_count = 1
        for i in range(1, len(password)):
            if ord(password[i]) == ord(password[i-1]) + 1:
                consecutive_count += 1
                if consecutive_count > self.requirements.max_consecutive_chars:
                    return False
            else:
                consecutive_count = 1
        
        return True
    
    def _check_repeated_chars(self, password: str) -> bool:
        """Check for repeated characters"""
        if self.requirements.max_repeated_chars <= 0:
            return True
        
        char_counts = {}
        for char in password:
            char_counts[char] = char_counts.get(char, 0) + 1
            if char_counts[char] > self.requirements.max_repeated_chars:
                return False
        
        return True
    
    def _check_forbidden_patterns(self, password: str) -> bool:
        """Check for forbidden patterns"""
        if not self.requirements.forbidden_patterns:
            return True
        
        password_lower = password.lower()
        for pattern in self.requirements.forbidden_patterns:
            if re.search(pattern.lower(), password_lower):
                return False
        
        return True
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy"""
        if not password:
            return 0.0
        
        # Count character types
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digits = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        
        # Calculate character set size
        charset_size = 0
        if has_lower:
            charset_size += 26
        if has_upper:
            charset_size += 26
        if has_digits:
            charset_size += 10
        if has_special:
            charset_size += 32  # Approximate special character count
        
        # Calculate entropy
        entropy = len(password) * (charset_size ** 0.5)
        return entropy
    
    def _determine_strength(self, zxcvbn_result: Dict, entropy: float, requirements_met: Dict[str, bool]) -> Tuple[PasswordStrength, int]:
        """Determine password strength and score"""
        zxcvbn_score = zxcvbn_result['score']
        entropy_score = min(entropy / 10, 10)  # Normalize entropy to 0-10 scale
        
        # Calculate overall score (0-100)
        score = int((zxcvbn_score * 20) + (entropy_score * 10))
        score = min(score, 100)
        
        # Determine strength based on score and requirements
        if score < 20 or not all(requirements_met.values()):
            return PasswordStrength.VERY_WEAK, score
        elif score < 40:
            return PasswordStrength.WEAK, score
        elif score < 60:
            return PasswordStrength.FAIR, score
        elif score < 80:
            return PasswordStrength.GOOD, score
        elif score < 95:
            return PasswordStrength.STRONG, score
        else:
            return PasswordStrength.VERY_STRONG, score
    
    def _generate_feedback(self, password: str, requirements_met: Dict[str, bool], zxcvbn_result: Dict) -> Tuple[List[str], List[str]]:
        """Generate feedback and suggestions"""
        feedback = []
        suggestions = []
        
        # Requirements feedback
        if not requirements_met.get('min_length', True):
            feedback.append(f"Password must be at least {self.requirements.min_length} characters long")
            suggestions.append(f"Add {self.requirements.min_length - len(password)} more characters")
        
        if not requirements_met.get('uppercase', True):
            feedback.append("Password must contain at least one uppercase letter")
            suggestions.append("Add uppercase letters (A-Z)")
        
        if not requirements_met.get('lowercase', True):
            feedback.append("Password must contain at least one lowercase letter")
            suggestions.append("Add lowercase letters (a-z)")
        
        if not requirements_met.get('digits', True):
            feedback.append("Password must contain at least one digit")
            suggestions.append("Add numbers (0-9)")
        
        if not requirements_met.get('special_chars', True):
            feedback.append("Password must contain at least one special character")
            suggestions.append("Add special characters (!@#$%^&*)")
        
        if not requirements_met.get('min_special_chars', True):
            feedback.append(f"Password must contain at least {self.requirements.min_special_chars} special characters")
            suggestions.append(f"Add {self.requirements.min_special_chars - len(re.findall(r'[!@#$%^&*(),.?\":{}|<>]', password))} more special characters")
        
        if not requirements_met.get('max_consecutive', True):
            feedback.append(f"Password contains more than {self.requirements.max_consecutive_chars} consecutive characters")
            suggestions.append("Avoid consecutive characters (abc, 123)")
        
        if not requirements_met.get('max_repeated', True):
            feedback.append(f"Password contains more than {self.requirements.max_repeated_chars} repeated characters")
            suggestions.append("Avoid repeating the same character")
        
        if not requirements_met.get('no_common_passwords', True):
            feedback.append("Password is too common")
            suggestions.append("Use a unique password that's not commonly used")
        
        # ZXCVBN feedback
        if zxcvbn_result.get('feedback', {}).get('warning'):
            feedback.append(zxcvbn_result['feedback']['warning'])
        
        if zxcvbn_result.get('feedback', {}).get('suggestions'):
            suggestions.extend(zxcvbn_result['feedback']['suggestions'])
        
        return feedback, suggestions
    
    def generate_strong_password(self, length: int = 16, include_special: bool = True) -> str:
        """Generate a strong password"""
        if length < 8:
            length = 8
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if include_special:
            chars += "!@#$%^&*(),.?\":{}|<>"
        
        # Generate password with guaranteed character types
        password = ""
        
        # Add at least one character from each required type
        if self.requirements.require_lowercase:
            password += secrets.choice("abcdefghijklmnopqrstuvwxyz")
        if self.requirements.require_uppercase:
            password += secrets.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if self.requirements.require_digits:
            password += secrets.choice("0123456789")
        if self.requirements.require_special_chars and include_special:
            password += secrets.choice("!@#$%^&*(),.?\":{}|<>")
        
        # Fill the rest with random characters
        remaining_length = length - len(password)
        for _ in range(remaining_length):
            password += secrets.choice(chars)
        
        # Shuffle the password
        password_list = list(password)
        secrets.SystemRandom().shuffle(password_list)
        password = ''.join(password_list)
        
        return password
    
    def get_strength_color(self, strength: PasswordStrength) -> str:
        """Get color code for password strength"""
        colors = {
            PasswordStrength.VERY_WEAK: "#ff0000",
            PasswordStrength.WEAK: "#ff6600",
            PasswordStrength.FAIR: "#ffcc00",
            PasswordStrength.GOOD: "#99cc00",
            PasswordStrength.STRONG: "#66cc00",
            PasswordStrength.VERY_STRONG: "#00cc00"
        }
        return colors.get(strength, "#ff0000")
    
    def get_strength_label(self, strength: PasswordStrength) -> str:
        """Get human-readable label for password strength"""
        labels = {
            PasswordStrength.VERY_WEAK: "Very Weak",
            PasswordStrength.WEAK: "Weak",
            PasswordStrength.FAIR: "Fair",
            PasswordStrength.GOOD: "Good",
            PasswordStrength.STRONG: "Strong",
            PasswordStrength.VERY_STRONG: "Very Strong"
        }
        return labels.get(strength, "Unknown")

# Default validator instance
default_validator = PasswordValidator()

def validate_password_strength(password: str, user_info: Optional[Dict] = None) -> PasswordValidationResult:
    """Convenience function for password validation"""
    return default_validator.validate_password(password, user_info)

def generate_strong_password(length: int = 16, include_special: bool = True) -> str:
    """Convenience function for password generation"""
    return default_validator.generate_strong_password(length, include_special)
