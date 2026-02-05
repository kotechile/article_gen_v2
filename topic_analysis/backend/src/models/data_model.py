"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
DataModel Entity

This module defines the DataModel entity for representing business entities
stored and retrieved through Supabase database tables.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import uuid

class DataModel(BaseModel):
    """
    DataModel entity representing business entities in the database.
    
    This model defines the structure and metadata for business entities
    stored in Supabase database tables.
    """
    
    # Primary identifier
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique model identifier")
    
    # Model identification
    model_name: str = Field(..., description="Name of the business entity")
    table_name: str = Field(..., description="Corresponding Supabase table name")
    schema_version: str = Field(..., description="Schema version identifier")
    
    # Model definition
    fields: Dict[str, Any] = Field(..., description="Model field definitions")
    relationships: Optional[Dict[str, Any]] = Field(default=None, description="Model relationships")
    validation_rules: Optional[Dict[str, Any]] = Field(default=None, description="Field validation rules")
    
    # Database configuration
    indexes: Optional[Dict[str, Any]] = Field(default=None, description="Database indexes")
    rls_policies: Optional[Dict[str, Any]] = Field(default=None, description="Row Level Security policies")
    
    # Model state
    is_active: bool = Field(default=True, description="Whether model is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Model creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    created_by: str = Field(..., description="User who created the model")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v or not v.strip():
            raise ValueError('Model name must be non-empty')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Model name must contain only alphanumeric characters, underscores, and hyphens')
        return v.strip()
    
    @validator('table_name')
    def validate_table_name(cls, v):
        """Validate table name format."""
        if not v or not v.strip():
            raise ValueError('Table name must be non-empty')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Table name must contain only alphanumeric characters, underscores, and hyphens')
        return v.strip()
    
    @validator('schema_version')
    def validate_schema_version(cls, v):
        """Validate schema version format."""
        if not v or not v.strip():
            raise ValueError('Schema version must be non-empty')
        # Basic semantic versioning validation
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError('Schema version must follow semantic versioning (e.g., 1.0.0)')
        for part in parts:
            if not part.isdigit():
                raise ValueError('Schema version parts must be numeric')
        return v.strip()
    
    @validator('fields')
    def validate_fields(cls, v):
        """Validate fields structure."""
        if not v or not isinstance(v, dict):
            raise ValueError('Fields must be a non-empty dictionary')
        if not v:
            raise ValueError('Fields must contain at least one field definition')
        return v
    
    @validator('updated_at')
    def validate_updated_at(cls, v, values):
        """Validate updated_at timestamp."""
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('updated_at must be after created_at')
        return v
    
    def update_model(self, fields: Optional[Dict[str, Any]] = None, 
                    relationships: Optional[Dict[str, Any]] = None,
                    validation_rules: Optional[Dict[str, Any]] = None,
                    indexes: Optional[Dict[str, Any]] = None,
                    rls_policies: Optional[Dict[str, Any]] = None) -> None:
        """Update model definition."""
        if fields is not None:
            self.fields = fields
        if relationships is not None:
            self.relationships = relationships
        if validation_rules is not None:
            self.validation_rules = validation_rules
        if indexes is not None:
            self.indexes = indexes
        if rls_policies is not None:
            self.rls_policies = rls_policies
        
        self.updated_at = datetime.utcnow()
    
    def deprecate(self) -> None:
        """Deprecate the model."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def reactivate(self) -> None:
        """Reactivate the model."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def get_field_definition(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get definition for a specific field."""
        return self.fields.get(field_name)
    
    def add_field(self, field_name: str, field_definition: Dict[str, Any]) -> None:
        """Add a new field to the model."""
        self.fields[field_name] = field_definition
        self.updated_at = datetime.utcnow()
    
    def remove_field(self, field_name: str) -> bool:
        """Remove a field from the model."""
        if field_name in self.fields:
            del self.fields[field_name]
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_relationships(self) -> Dict[str, Any]:
        """Get model relationships."""
        return self.relationships or {}
    
    def add_relationship(self, relationship_name: str, relationship_definition: Dict[str, Any]) -> None:
        """Add a new relationship to the model."""
        if self.relationships is None:
            self.relationships = {}
        self.relationships[relationship_name] = relationship_definition
        self.updated_at = datetime.utcnow()
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules."""
        return self.validation_rules or {}
    
    def add_validation_rule(self, field_name: str, rule: Dict[str, Any]) -> None:
        """Add validation rule for a field."""
        if self.validation_rules is None:
            self.validation_rules = {}
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        self.validation_rules[field_name].append(rule)
        self.updated_at = datetime.utcnow()
    
    def get_indexes(self) -> Dict[str, Any]:
        """Get database indexes."""
        return self.indexes or {}
    
    def add_index(self, index_name: str, index_definition: Dict[str, Any]) -> None:
        """Add a new index to the model."""
        if self.indexes is None:
            self.indexes = {}
        self.indexes[index_name] = index_definition
        self.updated_at = datetime.utcnow()
    
    def get_rls_policies(self) -> Dict[str, Any]:
        """Get Row Level Security policies."""
        return self.rls_policies or {}
    
    def add_rls_policy(self, policy_name: str, policy_definition: Dict[str, Any]) -> None:
        """Add a new RLS policy to the model."""
        if self.rls_policies is None:
            self.rls_policies = {}
        self.rls_policies[policy_name] = policy_definition
        self.updated_at = datetime.utcnow()
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata for monitoring."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "table_name": self.table_name,
            "schema_version": self.schema_version,
            "is_active": self.is_active,
            "field_count": len(self.fields),
            "relationship_count": len(self.relationships) if self.relationships else 0,
            "index_count": len(self.indexes) if self.indexes else 0,
            "rls_policy_count": len(self.rls_policies) if self.rls_policies else 0,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "table_name": self.table_name,
            "schema_version": self.schema_version,
            "fields": self.fields,
            "relationships": self.relationships,
            "validation_rules": self.validation_rules,
            "indexes": self.indexes,
            "rls_policies": self.rls_policies,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }
    
    @classmethod
    def create_model(cls, model_name: str, table_name: str, fields: Dict[str, Any], 
                    created_by: str, schema_version: str = "1.0.0") -> 'DataModel':
        """Create a new data model."""
        return cls(
            model_name=model_name,
            table_name=table_name,
            fields=fields,
            created_by=created_by,
            schema_version=schema_version
        )
    
    @classmethod
    def create_user_model(cls, created_by: str) -> 'DataModel':
        """Create a user model."""
        return cls(
            model_name="User",
            table_name="users",
            fields={
                "id": {"type": "uuid", "primary_key": True},
                "email": {"type": "varchar", "unique": True, "not_null": True},
                "name": {"type": "varchar", "not_null": True},
                "status": {"type": "varchar", "default": "active"},
                "created_at": {"type": "timestamp", "default": "now()"},
                "updated_at": {"type": "timestamp", "default": "now()"}
            },
            created_by=created_by,
            schema_version="1.0.0"
        )
    
    @classmethod
    def create_trend_analysis_model(cls, created_by: str) -> 'DataModel':
        """Create a trend analysis model."""
        return cls(
            model_name="TrendAnalysis",
            table_name="trend_analysis",
            fields={
                "id": {"type": "uuid", "primary_key": True},
                "user_id": {"type": "uuid", "foreign_key": "users.id"},
                "keyword": {"type": "varchar", "not_null": True},
                "trend_data": {"type": "jsonb"},
                "status": {"type": "varchar", "default": "pending"},
                "created_at": {"type": "timestamp", "default": "now()"},
                "updated_at": {"type": "timestamp", "default": "now()"}
            },
            created_by=created_by,
            schema_version="1.0.0"
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"DataModel(name={self.model_name}, table={self.table_name}, active={self.is_active})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DataModel("
            f"model_id='{self.model_id}', "
            f"model_name='{self.model_name}', "
            f"table_name='{self.table_name}', "
            f"schema_version='{self.schema_version}', "
            f"is_active={self.is_active}"
            f")"
        )