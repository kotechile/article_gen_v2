"""
Export Integration API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ExportFormat(str, Enum):
    """Export format enumeration"""
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    PDF = "pdf"
    XML = "xml"
    HTML = "html"

class ExportStatus(str, Enum):
    """Export status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExportType(str, Enum):
    """Export type enumeration"""
    AFFILIATE_RESEARCH = "affiliate_research"
    TREND_ANALYSIS = "trend_analysis"
    KEYWORD_DATA = "keyword_data"
    CONTENT_IDEAS = "content_ideas"
    SOFTWARE_SOLUTIONS = "software_solutions"
    CONTENT_CALENDAR = "content_calendar"
    USER_DATA = "user_data"
    ANALYTICS = "analytics"

class ExportRequest(BaseModel):
    """Request schema for creating export"""
    export_type: ExportType = Field(..., description="Type of data to export")
    format: ExportFormat = Field(..., description="Export format")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    fields: Optional[List[str]] = Field(None, description="Fields to include")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    include_metadata: Optional[bool] = Field(True, description="Include metadata")
    include_charts: Optional[bool] = Field(False, description="Include charts/images")
    compression: Optional[bool] = Field(False, description="Compress export file")
    password_protect: Optional[bool] = Field(False, description="Password protect file")
    custom_filename: Optional[str] = Field(None, description="Custom filename")
    
    @validator('fields')
    def validate_fields(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError('Fields list cannot be empty')
        return v

class ExportResponse(BaseModel):
    """Response schema for export"""
    id: str = Field(..., description="Export ID")
    user_id: int = Field(..., description="User ID")
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    status: ExportStatus = Field(..., description="Export status")
    file_url: Optional[str] = Field(None, description="Download URL")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    record_count: Optional[int] = Field(None, ge=0, description="Number of records exported")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Applied filters")
    fields_included: Optional[List[str]] = Field(None, description="Included fields")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Export metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Export creation date")
    completed_at: Optional[datetime] = Field(None, description="Export completion date")
    expires_at: Optional[datetime] = Field(None, description="Download expiration date")

class ExportListResponse(BaseModel):
    """Response schema for export list"""
    exports: List[ExportResponse] = Field(..., description="List of exports")
    total: int = Field(..., description="Total number of exports")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ExportTemplateResponse(BaseModel):
    """Response schema for export template"""
    id: int = Field(..., description="Template ID")
    user_id: int = Field(..., description="User ID")
    name: str = Field(..., description="Template name")
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    description: Optional[str] = Field(None, description="Template description")
    template_config: Dict[str, Any] = Field(..., description="Template configuration")
    is_public: bool = Field(..., description="Is template public")
    usage_count: int = Field(..., description="Usage count")
    created_at: datetime = Field(..., description="Template creation date")
    updated_at: datetime = Field(..., description="Last update date")

class ExportTemplateListResponse(BaseModel):
    """Response schema for export template list"""
    templates: List[ExportTemplateResponse] = Field(..., description="List of templates")
    total: int = Field(..., description="Total number of templates")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ExportTemplateCreateRequest(BaseModel):
    """Request schema for creating export template"""
    name: str = Field(..., min_length=1, max_length=200, description="Template name")
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    description: Optional[str] = Field(None, description="Template description")
    template_config: Dict[str, Any] = Field(..., description="Template configuration")
    is_public: Optional[bool] = Field(False, description="Make template public")

class ExportTemplateUpdateRequest(BaseModel):
    """Request schema for updating export template"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated name")
    description: Optional[str] = Field(None, description="Updated description")
    template_config: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")
    is_public: Optional[bool] = Field(None, description="Updated public status")

class ExportScheduleRequest(BaseModel):
    """Request schema for scheduling export"""
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    schedule: str = Field(..., description="Cron schedule expression")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    fields: Optional[List[str]] = Field(None, description="Fields to include")
    email_notification: Optional[bool] = Field(True, description="Send email notification")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notification")
    enabled: Optional[bool] = Field(True, description="Enable schedule")

class ExportScheduleResponse(BaseModel):
    """Response schema for export schedule"""
    id: int = Field(..., description="Schedule ID")
    user_id: int = Field(..., description="User ID")
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    schedule: str = Field(..., description="Cron schedule expression")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    fields: Optional[List[str]] = Field(None, description="Fields to include")
    email_notification: bool = Field(..., description="Email notification enabled")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    enabled: bool = Field(..., description="Schedule enabled")
    last_run: Optional[datetime] = Field(None, description="Last run date")
    next_run: Optional[datetime] = Field(None, description="Next run date")
    created_at: datetime = Field(..., description="Schedule creation date")
    updated_at: datetime = Field(..., description="Last update date")

class ExportScheduleListResponse(BaseModel):
    """Response schema for export schedule list"""
    schedules: List[ExportScheduleResponse] = Field(..., description="List of schedules")
    total: int = Field(..., description="Total number of schedules")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ExportStatsResponse(BaseModel):
    """Response schema for export statistics"""
    total_exports: int = Field(..., description="Total exports")
    exports_by_type: List[Dict[str, Any]] = Field(..., description="Exports by type")
    exports_by_format: List[Dict[str, Any]] = Field(..., description="Exports by format")
    total_file_size: int = Field(..., description="Total file size in bytes")
    average_file_size: float = Field(..., description="Average file size")
    total_records_exported: int = Field(..., description="Total records exported")
    average_records_per_export: float = Field(..., description="Average records per export")
    success_rate: float = Field(..., description="Export success rate")
    active_schedules: int = Field(..., description="Active schedules")
    last_updated: datetime = Field(..., description="Last database update")

class ExportValidationRequest(BaseModel):
    """Request schema for export validation"""
    export_type: ExportType = Field(..., description="Export type")
    format: ExportFormat = Field(..., description="Export format")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    fields: Optional[List[str]] = Field(None, description="Fields to include")

class ExportValidationResponse(BaseModel):
    """Response schema for export validation"""
    is_valid: bool = Field(..., description="Is export configuration valid")
    estimated_records: int = Field(..., description="Estimated number of records")
    estimated_file_size: int = Field(..., description="Estimated file size in bytes")
    available_fields: List[str] = Field(..., description="Available fields")
    warnings: List[str] = Field(..., description="Validation warnings")
    errors: List[str] = Field(..., description="Validation errors")
    recommendations: List[str] = Field(..., description="Recommendations")

class ExportStatusResponse(BaseModel):
    """Response schema for export status"""
    export_id: str = Field(..., description="Export ID")
    status: str = Field(..., description="Export status")
    progress: int = Field(..., ge=0, le=100, description="Export progress percentage")
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(..., description="Export creation date")
    completed_at: Optional[datetime] = Field(None, description="Export completion date")
