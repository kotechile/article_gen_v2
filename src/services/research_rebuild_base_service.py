"""
Shared helpers for the research rebuild service layer.

These services are intentionally thin scaffolds. They provide a stable place
for persistence and status transitions while the end-to-end rebuild is wired
up incrementally behind feature flags.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from .supabase_service import SupabaseService

logger = logging.getLogger(__name__)


class ResearchRebuildBaseService:
    """Base class for the new research rebuild services."""

    table_name: str = ""

    def __init__(self, supabase_service: Optional[SupabaseService] = None):
        self.supabase_service = supabase_service or SupabaseService()

    async def list_records(
        self,
        *,
        user_id: UUID,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch records for the current user."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")

        return await self.supabase_service.get_by_filters(
            self.table_name,
            filters=filters or {},
            user_id=user_id,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )

    async def get_record(self, *, record_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Fetch a single record by id."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")
        return await self.supabase_service.get_by_id(self.table_name, record_id, user_id)

    async def create_record(self, *, user_id: UUID, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new record in the service table."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")
        return await self.supabase_service.create(self.table_name, data=data, user_id=user_id)

    async def bulk_create_records(
        self,
        *,
        user_id: UUID,
        data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bulk create records in the service table."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")
        if not data_list:
            return []
        return await self.supabase_service.bulk_create(self.table_name, data_list=data_list, user_id=user_id)

    async def update_record(
        self,
        *,
        record_id: UUID,
        user_id: UUID,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update a record in the service table."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")
        payload = dict(data)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        return await self.supabase_service.update(self.table_name, record_id, data=payload, user_id=user_id)

    async def delete_record(self, *, record_id: UUID, user_id: UUID) -> bool:
        """Delete a record from the service table."""
        if not self.table_name:
            raise ValueError("table_name must be defined on subclass")
        return await self.supabase_service.delete(self.table_name, record_id, user_id)

    @staticmethod
    def compute_validation_expiry(*, ttl_days: int) -> str:
        """Compute an ISO timestamp for validation expiry."""
        return (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    @staticmethod
    def normalize_reason_tags(reason_tags: Optional[List[str]]) -> List[str]:
        """Normalize free-form reason tags into a compact lowercase list."""
        values: List[str] = []
        for value in reason_tags or []:
            cleaned = str(value or "").strip().lower()
            if cleaned and cleaned not in values:
                values.append(cleaned)
        return values
