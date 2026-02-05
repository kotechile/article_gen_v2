"""
Audit Service
Provides comprehensive audit logging for compliance and security
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache

logger = structlog.get_logger()

class AuditEventType(Enum):
    """Audit event types"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTRATION = "user_registration"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    CONTENT_CREATE = "content_create"
    CONTENT_UPDATE = "content_update"
    CONTENT_DELETE = "content_delete"
    TREND_ANALYSIS = "trend_analysis"
    AFFILIATE_RESEARCH = "affiliate_research"
    KEYWORD_UPLOAD = "keyword_upload"
    FILE_UPLOAD = "file_upload"
    API_CALL = "api_call"
    DATA_EXPORT = "data_export"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"
    ADMIN_ACTION = "admin_action"

class AuditService:
    """Service for comprehensive audit logging"""
    
    def __init__(self):
        self.db = get_supabase_db()
        self.redis = cache
        self.audit_ttl = 30 * 24 * 3600  # 30 days cache TTL
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        action: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        severity: str = "info"
    ) -> Dict[str, Any]:
        """
        Log an audit event
        
        Args:
            event_type: Type of audit event
            user_id: User ID (optional for system events)
            action: Action performed
            resource_id: Optional resource ID
            resource_type: Optional resource type
            details: Optional additional details
            ip_address: Optional IP address
            user_agent: Optional user agent
            severity: Event severity (info, warning, error, critical)
            
        Returns:
            Dict containing audit log results
        """
        try:
            logger.info("Logging audit event", 
                       event_type=event_type.value,
                       user_id=user_id,
                       action=action,
                       severity=severity)
            
            # Create audit event
            audit_event = {
                "event_type": event_type.value,
                "user_id": user_id,
                "action": action,
                "resource_id": resource_id,
                "resource_type": resource_type,
                "details": details or {},
                "ip_address": ip_address,
                "user_agent": user_agent,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "id": f"audit_{datetime.utcnow().timestamp()}_{user_id or 'system'}"
            }
            
            # Store in database
            await self._store_audit_event(audit_event)
            
            # Cache for quick access
            await self._cache_audit_event(audit_event)
            
            # Check for security events
            if severity in ["error", "critical"] or event_type == AuditEventType.SECURITY_EVENT:
                await self._handle_security_event(audit_event)
            
            logger.info("Audit event logged successfully", 
                       event_id=audit_event["id"],
                       event_type=event_type.value)
            
            return {
                "success": True,
                "event_id": audit_event["id"],
                "event_type": event_type.value,
                "timestamp": audit_event["timestamp"]
            }
            
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
            raise
    
    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get audit logs with filtering
        
        Args:
            user_id: Optional user ID filter
            event_type: Optional event type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            severity: Optional severity filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dict containing audit logs
        """
        try:
            logger.info("Getting audit logs", 
                       user_id=user_id,
                       event_type=event_type.value if event_type else None,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None,
                       severity=severity,
                       limit=limit,
                       offset=offset)
            
            # Build query filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if event_type:
                filters["event_type"] = event_type.value
            if start_date:
                filters["start_date"] = start_date.isoformat()
            if end_date:
                filters["end_date"] = end_date.isoformat()
            if severity:
                filters["severity"] = severity
            
            # Get audit logs from database
            audit_logs = await self.db.get_audit_logs(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Get total count for pagination
            total_count = await self.db.get_audit_logs_count(filters)
            
            logger.info("Audit logs retrieved", 
                       count=len(audit_logs),
                       total_count=total_count)
            
            return {
                "success": True,
                "audit_logs": audit_logs,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
            
        except Exception as e:
            logger.error("Failed to get audit logs", error=str(e))
            raise
    
    async def get_user_activity_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get user activity summary
        
        Args:
            user_id: User ID
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dict containing user activity summary
        """
        try:
            logger.info("Getting user activity summary", 
                       user_id=user_id,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get user activity data
            activity_data = await self.db.get_user_activity_summary(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate activity metrics
            metrics = await self._calculate_activity_metrics(activity_data)
            
            # Generate activity insights
            insights = await self._generate_activity_insights(activity_data, metrics)
            
            logger.info("User activity summary calculated", 
                       user_id=user_id,
                       total_events=metrics.get("total_events", 0),
                       insights_count=len(insights))
            
            return {
                "success": True,
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "metrics": metrics,
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get user activity summary", error=str(e))
            raise
    
    async def get_security_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get security events
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            severity: Optional severity filter
            
        Returns:
            Dict containing security events
        """
        try:
            logger.info("Getting security events", 
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None,
                       severity=severity)
            
            # Get security events
            security_events = await self.get_audit_logs(
                event_type=AuditEventType.SECURITY_EVENT,
                start_date=start_date,
                end_date=end_date,
                severity=severity,
                limit=1000
            )
            
            # Analyze security patterns
            patterns = await self._analyze_security_patterns(security_events["audit_logs"])
            
            # Generate security recommendations
            recommendations = await self._generate_security_recommendations(patterns)
            
            logger.info("Security events analyzed", 
                       events_count=len(security_events["audit_logs"]),
                       patterns_count=len(patterns),
                       recommendations_count=len(recommendations))
            
            return {
                "success": True,
                "security_events": security_events["audit_logs"],
                "patterns": patterns,
                "recommendations": recommendations,
                "total_count": security_events["total_count"],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get security events", error=str(e))
            raise
    
    async def export_audit_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export audit logs
        
        Args:
            user_id: Optional user ID filter
            event_type: Optional event type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            format: Export format (json, csv)
            
        Returns:
            Dict containing export results
        """
        try:
            logger.info("Exporting audit logs", 
                       user_id=user_id,
                       event_type=event_type.value if event_type else None,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None,
                       format=format)
            
            # Get all audit logs (no limit for export)
            audit_logs = await self.get_audit_logs(
                user_id=user_id,
                event_type=event_type,
                start_date=start_date,
                end_date=end_date,
                limit=10000,  # Large limit for export
                offset=0
            )
            
            # Generate export file
            export_result = await self._generate_audit_export(
                audit_logs=audit_logs["audit_logs"],
                format=format
            )
            
            logger.info("Audit logs exported", 
                       format=format,
                       records_count=len(audit_logs["audit_logs"]),
                       file_size=export_result.get("file_size", 0))
            
            return {
                "success": True,
                "format": format,
                "records_count": len(audit_logs["audit_logs"]),
                "export_result": export_result,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export audit logs", error=str(e))
            raise
    
    async def _store_audit_event(self, audit_event: Dict[str, Any]) -> None:
        """Store audit event in database"""
        try:
            await self.db.create_audit_log(audit_event)
        except Exception as e:
            logger.error("Failed to store audit event", error=str(e))
            raise
    
    async def _cache_audit_event(self, audit_event: Dict[str, Any]) -> None:
        """Cache audit event for quick access"""
        try:
            cache_key = f"audit:{audit_event['id']}"
            await self.redis.setex(
                cache_key,
                self.audit_ttl,
                json.dumps(audit_event)
            )
        except Exception as e:
            logger.warning("Failed to cache audit event", error=str(e))
    
    async def _handle_security_event(self, audit_event: Dict[str, Any]) -> None:
        """Handle security events"""
        try:
            # Log security event
            logger.warning("Security event detected", 
                          event_id=audit_event["id"],
                          event_type=audit_event["event_type"],
                          severity=audit_event["severity"],
                          user_id=audit_event["user_id"])
            
            # Store in security events table
            await self.db.create_security_event(audit_event)
            
            # Send alert if critical
            if audit_event["severity"] == "critical":
                await self._send_security_alert(audit_event)
                
        except Exception as e:
            logger.error("Failed to handle security event", error=str(e))
    
    async def _send_security_alert(self, audit_event: Dict[str, Any]) -> None:
        """Send security alert for critical events"""
        try:
            # This would integrate with notification service
            logger.critical("Security alert triggered", 
                          event_id=audit_event["id"],
                          event_type=audit_event["event_type"],
                          user_id=audit_event["user_id"])
        except Exception as e:
            logger.error("Failed to send security alert", error=str(e))
    
    async def _calculate_activity_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate activity metrics"""
        metrics = {
            "total_events": len(activity_data.get("events", [])),
            "unique_actions": len(set(event.get("action") for event in activity_data.get("events", []))),
            "event_types": {},
            "daily_activity": {},
            "hourly_activity": {},
            "most_active_hours": [],
            "activity_trend": "stable"
        }
        
        # Count event types
        for event in activity_data.get("events", []):
            event_type = event.get("event_type", "unknown")
            metrics["event_types"][event_type] = metrics["event_types"].get(event_type, 0) + 1
        
        # Calculate daily activity
        for event in activity_data.get("events", []):
            date = event.get("timestamp", "").split("T")[0]
            metrics["daily_activity"][date] = metrics["daily_activity"].get(date, 0) + 1
        
        # Calculate hourly activity
        for event in activity_data.get("events", []):
            try:
                hour = datetime.fromisoformat(event.get("timestamp", "")).hour
                metrics["hourly_activity"][hour] = metrics["hourly_activity"].get(hour, 0) + 1
            except:
                pass
        
        # Find most active hours
        if metrics["hourly_activity"]:
            sorted_hours = sorted(metrics["hourly_activity"].items(), key=lambda x: x[1], reverse=True)
            metrics["most_active_hours"] = [hour for hour, count in sorted_hours[:3]]
        
        return metrics
    
    async def _generate_activity_insights(
        self,
        activity_data: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate activity insights"""
        insights = []
        
        # Activity level insights
        total_events = metrics.get("total_events", 0)
        if total_events > 100:
            insights.append({
                "type": "activity",
                "title": "High Activity",
                "description": f"User has {total_events} events in the period",
                "recommendation": "Consider reviewing activity patterns"
            })
        elif total_events < 10:
            insights.append({
                "type": "activity",
                "title": "Low Activity",
                "description": f"User has only {total_events} events in the period",
                "recommendation": "Consider increasing platform usage"
            })
        
        # Time pattern insights
        most_active_hours = metrics.get("most_active_hours", [])
        if most_active_hours:
            insights.append({
                "type": "pattern",
                "title": "Activity Patterns",
                "description": f"Most active during hours: {', '.join(map(str, most_active_hours))}",
                "recommendation": "Schedule important tasks during peak hours"
            })
        
        return insights
    
    async def _analyze_security_patterns(self, security_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze security patterns"""
        patterns = []
        
        # Group events by type
        event_types = {}
        for event in security_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Analyze patterns
        for event_type, events in event_types.items():
            if len(events) > 5:
                patterns.append({
                    "type": "frequency",
                    "event_type": event_type,
                    "count": len(events),
                    "description": f"High frequency of {event_type} events",
                    "severity": "warning"
                })
        
        return patterns
    
    async def _generate_security_recommendations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate security recommendations"""
        recommendations = []
        
        for pattern in patterns:
            if pattern["type"] == "frequency":
                recommendations.append({
                    "type": "security",
                    "title": "Review Security Events",
                    "description": f"High frequency of {pattern['event_type']} events detected",
                    "action": "Investigate and implement additional security measures",
                    "priority": "high" if pattern["severity"] == "warning" else "medium"
                })
        
        return recommendations
    
    async def _generate_audit_export(
        self,
        audit_logs: List[Dict[str, Any]],
        format: str
    ) -> Dict[str, Any]:
        """Generate audit export file"""
        try:
            if format == "json":
                return {
                    "file_type": "json",
                    "file_size": len(json.dumps(audit_logs)),
                    "download_url": f"/exports/audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                }
            elif format == "csv":
                # Convert to CSV format
                return {
                    "file_type": "csv",
                    "file_size": 0,  # Would be calculated
                    "download_url": f"/exports/audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error("Failed to generate audit export", error=str(e))
            raise

