"""
API Key Service
Handles encrypted external API key management
"""

import os
import base64
from cryptography.fernet import Fernet
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from ..models.api_key import APIKey

logger = structlog.get_logger()

class APIKeyService:
    """Service for managing external API keys with encryption"""
    
    def __init__(self, db: Session):
        self.db = db
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for API keys"""
        key = os.getenv("API_KEY_ENCRYPTION_KEY")
        if not key:
            # Generate a new key if none exists
            key = Fernet.generate_key()
            logger.warning("Generated new API key encryption key. Set API_KEY_ENCRYPTION_KEY environment variable for production.")
        else:
            key = key.encode() if isinstance(key, str) else key
        return key
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage"""
        try:
            encrypted_key = self.cipher.encrypt(api_key.encode())
            return base64.b64encode(encrypted_key).decode()
        except Exception as e:
            logger.error("Failed to encrypt API key", error=str(e))
            raise
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key for use"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_key.encode())
            decrypted_key = self.cipher.decrypt(encrypted_bytes)
            return decrypted_key.decode()
        except Exception as e:
            logger.error("Failed to decrypt API key", error=str(e))
            raise
    
    def create_api_key(
        self, 
        user_id: str, 
        service_name: str, 
        api_key: str
    ) -> Dict[str, Any]:
        """
        Create a new API key for a user and service
        
        Args:
            user_id: User ID
            service_name: Name of the service (linkup, semrush, ahrefs, google_trends)
            api_key: The API key to store
            
        Returns:
            Dict containing the created API key data
        """
        try:
            # Validate service name
            valid_services = ['linkup', 'semrush', 'ahrefs', 'google_trends']
            if service_name not in valid_services:
                raise ValueError(f"Invalid service name. Must be one of: {valid_services}")
            
            # Check if user already has an active key for this service
            existing_key = self.db.query(APIKey).filter(
                APIKey.user_id == user_id,
                APIKey.service_name == service_name,
                APIKey.is_active == True
            ).first()
            
            if existing_key:
                # Deactivate existing key
                existing_key.is_active = False
                existing_key.updated_at = datetime.utcnow()
            
            # Encrypt the API key
            encrypted_key = self.encrypt_api_key(api_key)
            
            # Create new API key
            new_key = APIKey(
                user_id=user_id,
                service_name=service_name,
                api_key=encrypted_key,
                encrypted=True,
                is_active=True
            )
            
            self.db.add(new_key)
            self.db.commit()
            self.db.refresh(new_key)
            
            logger.info("API key created", 
                       user_id=user_id, 
                       service_name=service_name,
                       key_id=new_key.id)
            
            return new_key.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to create API key", 
                        user_id=user_id, 
                        service_name=service_name,
                        error=str(e))
            raise
    
    def get_api_key(self, user_id: str, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get active API key for user and service
        
        Args:
            user_id: User ID
            service_name: Name of the service
            
        Returns:
            API key data if found, None otherwise
        """
        try:
            api_key = self.db.query(APIKey).filter(
                APIKey.user_id == user_id,
                APIKey.service_name == service_name,
                APIKey.is_active == True
            ).first()
            
            if not api_key:
                return None
            
            return api_key.to_dict()
            
        except Exception as e:
            logger.error("Failed to get API key", 
                        user_id=user_id, 
                        service_name=service_name,
                        error=str(e))
            raise
    
    def get_decrypted_api_key(self, user_id: str, service_name: str) -> Optional[str]:
        """
        Get decrypted API key for use in external service calls
        
        Args:
            user_id: User ID
            service_name: Name of the service
            
        Returns:
            Decrypted API key if found, None otherwise
        """
        try:
            api_key = self.db.query(APIKey).filter(
                APIKey.user_id == user_id,
                APIKey.service_name == service_name,
                APIKey.is_active == True
            ).first()
            
            if not api_key:
                return None
            
            if api_key.encrypted:
                return self.decrypt_api_key(api_key.api_key)
            else:
                return api_key.api_key
                
        except Exception as e:
            logger.error("Failed to get decrypted API key", 
                        user_id=user_id, 
                        service_name=service_name,
                        error=str(e))
            raise
    
    def update_api_key(
        self, 
        key_id: str, 
        user_id: str, 
        api_key: str = None,
        is_active: bool = None
    ) -> Dict[str, Any]:
        """
        Update API key
        
        Args:
            key_id: API key ID
            user_id: User ID
            api_key: New API key (optional)
            is_active: New active status (optional)
            
        Returns:
            Updated API key data
        """
        try:
            api_key_obj = self.db.query(APIKey).filter(
                APIKey.id == key_id,
                APIKey.user_id == user_id
            ).first()
            
            if not api_key_obj:
                raise ValueError("API key not found")
            
            if api_key is not None:
                encrypted_key = self.encrypt_api_key(api_key)
                api_key_obj.api_key = encrypted_key
                api_key_obj.encrypted = True
            
            if is_active is not None:
                api_key_obj.is_active = is_active
            
            api_key_obj.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(api_key_obj)
            
            logger.info("API key updated", 
                       key_id=key_id, 
                       user_id=user_id)
            
            return api_key_obj.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to update API key", 
                        key_id=key_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Delete API key
        
        Args:
            key_id: API key ID
            user_id: User ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            api_key = self.db.query(APIKey).filter(
                APIKey.id == key_id,
                APIKey.user_id == user_id
            ).first()
            
            if not api_key:
                return False
            
            self.db.delete(api_key)
            self.db.commit()
            
            logger.info("API key deleted", 
                       key_id=key_id, 
                       user_id=user_id)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete API key", 
                        key_id=key_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def list_user_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all API keys for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of API key data
        """
        try:
            api_keys = self.db.query(APIKey).filter(
                APIKey.user_id == user_id
            ).order_by(APIKey.created_at.desc()).all()
            
            return [key.to_dict() for key in api_keys]
            
        except Exception as e:
            logger.error("Failed to list user API keys", 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def validate_api_key(self, user_id: str, service_name: str) -> bool:
        """
        Validate that user has a valid API key for the service
        
        Args:
            user_id: User ID
            service_name: Name of the service
            
        Returns:
            True if valid key exists, False otherwise
        """
        try:
            api_key = self.get_api_key(user_id, service_name)
            return api_key is not None
            
        except Exception as e:
            logger.error("Failed to validate API key", 
                        user_id=user_id, 
                        service_name=service_name,
                        error=str(e))
            return False
