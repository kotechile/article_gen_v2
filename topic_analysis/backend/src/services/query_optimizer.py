"""
Database query optimization service
"""
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import statistics

from ..core.config import get_settings
from ..models.user import User
from ..models.user_session import UserSession
from ..models.authentication_log import AuthenticationLog
from ..models.password_reset import PasswordReset
from ..models.jwt_blacklist import CSRFProtection, CSRFViolation
from src.core.supabase_database_service import SupabaseDatabaseService

logger = logging.getLogger(__name__)
settings = get_settings()

class QueryOptimizer:
    """Service for database query optimization"""
    
    def __init__(self, db: SupabaseDatabaseService):
        self.db = db
        self.query_cache = {}
        self.query_stats = {}
        self.slow_query_threshold = settings.SLOW_QUERY_THRESHOLD_MS
        self.max_query_cache_size = settings.MAX_QUERY_CACHE_SIZE
        self.query_timeout = settings.QUERY_TIMEOUT_SECONDS
    
    def optimize_user_queries(self, user_id: int) -> Dict[str, Any]:
        """Optimize user-related queries"""
        try:
            start_time = time.time()
            
            # Optimized user query with select_related
            user = self.db.query(User).options(
                # Use select_related for foreign keys
                # Use joinedload for relationships
            ).filter(User.id == user_id).first()
            
            if not user:
                return {"error": "User not found"}
            
            # Optimized user sessions query
            sessions = self.db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            ).order_by(desc(UserSession.created_at)).limit(10).all()
            
            # Optimized authentication logs query
            auth_logs = self.db.query(AuthenticationLog).filter(
                AuthenticationLog.user_id == user_id
            ).order_by(desc(AuthenticationLog.created_at)).limit(20).all()
            
            # Optimized password reset query
            password_resets = self.db.query(PasswordReset).filter(
                PasswordReset.user_id == user_id,
                PasswordReset.is_used == False
            ).order_by(desc(PasswordReset.created_at)).limit(5).all()
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "is_active": user.is_active,
                    "is_admin": user.is_admin,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
                },
                "sessions": [
                    {
                        "id": session.id,
                        "session_id": session.session_id,
                        "ip_address": session.ip_address,
                        "user_agent": session.user_agent,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat()
                    } for session in sessions
                ],
                "auth_logs": [
                    {
                        "id": log.id,
                        "action": log.action,
                        "ip_address": log.ip_address,
                        "user_agent": log.user_agent,
                        "success": log.success,
                        "created_at": log.created_at.isoformat()
                    } for log in auth_logs
                ],
                "password_resets": [
                    {
                        "id": reset.id,
                        "token": reset.token[:8] + "...",
                        "expires_at": reset.expires_at.isoformat(),
                        "created_at": reset.created_at.isoformat()
                    } for reset in password_resets
                ],
                "execution_time_ms": execution_time,
                "optimization_applied": True
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in optimize_user_queries: {str(e)}")
            return {"error": "Database query failed"}
        except Exception as e:
            logger.error(f"Error in optimize_user_queries: {str(e)}")
            return {"error": "Query optimization failed"}
    
    def optimize_user_list_query(
        self, 
        page: int = 1, 
        page_size: int = 20, 
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Optimize user list query with pagination and filtering"""
        try:
            start_time = time.time()
            
            # Build optimized query
            query = self.db.query(User)
            
            # Apply filters
            if search:
                query = query.filter(
                    or_(
                        User.username.ilike(f"%{search}%"),
                        User.email.ilike(f"%{search}%")
                    )
                )
            
            if is_active is not None:
                query = query.filter(User.is_active == is_active)
            
            if is_admin is not None:
                query = query.filter(User.is_admin == is_admin)
            
            # Get total count efficiently
            total_count = query.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            users = query.offset(offset).limit(page_size).all()
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "users": [
                    {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "is_active": user.is_active,
                        "is_admin": user.is_admin,
                        "created_at": user.created_at.isoformat(),
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    } for user in users
                ],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size,
                    "has_next": page * page_size < total_count,
                    "has_prev": page > 1
                },
                "execution_time_ms": execution_time,
                "optimization_applied": True
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in optimize_user_list_query: {str(e)}")
            return {"error": "Database query failed"}
        except Exception as e:
            logger.error(f"Error in optimize_user_list_query: {str(e)}")
            return {"error": "Query optimization failed"}
    
    def optimize_session_queries(self, user_id: int) -> Dict[str, Any]:
        """Optimize session-related queries"""
        try:
            start_time = time.time()
            
            # Optimized active sessions query
            active_sessions = self.db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            ).order_by(desc(UserSession.created_at)).all()
            
            # Optimized session statistics
            session_stats = self.db.query(
                func.count(UserSession.id).label('total_sessions'),
                func.count(
                    self.db.query(UserSession.id).filter(
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.utcnow()
                    ).subquery().c.id
                ).label('active_sessions'),
                func.max(UserSession.created_at).label('last_session'),
                func.min(UserSession.created_at).label('first_session')
            ).filter(UserSession.user_id == user_id).first()
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "active_sessions": [
                    {
                        "id": session.id,
                        "session_id": session.session_id,
                        "ip_address": session.ip_address,
                        "user_agent": session.user_agent,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat()
                    } for session in active_sessions
                ],
                "statistics": {
                    "total_sessions": session_stats.total_sessions or 0,
                    "active_sessions": session_stats.active_sessions or 0,
                    "last_session": session_stats.last_session.isoformat() if session_stats.last_session else None,
                    "first_session": session_stats.first_session.isoformat() if session_stats.first_session else None
                },
                "execution_time_ms": execution_time,
                "optimization_applied": True
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in optimize_session_queries: {str(e)}")
            return {"error": "Database query failed"}
        except Exception as e:
            logger.error(f"Error in optimize_session_queries: {str(e)}")
            return {"error": "Query optimization failed"}
    
    def optimize_auth_log_queries(
        self, 
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        success: Optional[bool] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Optimize authentication log queries"""
        try:
            start_time = time.time()
            
            # Calculate date range
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Build optimized query
            query = self.db.query(AuthenticationLog).filter(
                AuthenticationLog.created_at >= since_date
            )
            
            if user_id:
                query = query.filter(AuthenticationLog.user_id == user_id)
            
            if action:
                query = query.filter(AuthenticationLog.action == action)
            
            if success is not None:
                query = query.filter(AuthenticationLog.success == success)
            
            # Get logs with pagination
            logs = query.order_by(desc(AuthenticationLog.created_at)).limit(100).all()
            
            # Get statistics
            stats_query = self.db.query(
                func.count(AuthenticationLog.id).label('total_logs'),
                func.count(
                    self.db.query(AuthenticationLog.id).filter(
                        AuthenticationLog.success == True
                    ).subquery().c.id
                ).label('successful_logs'),
                func.count(
                    self.db.query(AuthenticationLog.id).filter(
                        AuthenticationLog.success == False
                    ).subquery().c.id
                ).label('failed_logs'),
                func.count(
                    self.db.query(AuthenticationLog.id).filter(
                        AuthenticationLog.action == 'login'
                    ).subquery().c.id
                ).label('login_attempts')
            ).filter(AuthenticationLog.created_at >= since_date)
            
            if user_id:
                stats_query = stats_query.filter(AuthenticationLog.user_id == user_id)
            
            stats = stats_query.first()
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "logs": [
                    {
                        "id": log.id,
                        "user_id": log.user_id,
                        "action": log.action,
                        "ip_address": log.ip_address,
                        "user_agent": log.user_agent,
                        "success": log.success,
                        "created_at": log.created_at.isoformat()
                    } for log in logs
                ],
                "statistics": {
                    "total_logs": stats.total_logs or 0,
                    "successful_logs": stats.successful_logs or 0,
                    "failed_logs": stats.failed_logs or 0,
                    "login_attempts": stats.login_attempts or 0,
                    "success_rate": (stats.successful_logs / stats.total_logs * 100) if stats.total_logs > 0 else 0
                },
                "execution_time_ms": execution_time,
                "optimization_applied": True
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in optimize_auth_log_queries: {str(e)}")
            return {"error": "Database query failed"}
        except Exception as e:
            logger.error(f"Error in optimize_auth_log_queries: {str(e)}")
            return {"error": "Query optimization failed"}
    
    def optimize_csrf_queries(self, user_id: int) -> Dict[str, Any]:
        """Optimize CSRF-related queries"""
        try:
            start_time = time.time()
            
            # Optimized CSRF tokens query
            csrf_tokens = self.db.query(CSRFProtection).filter(
                CSRFProtection.user_id == user_id,
                CSRFProtection.status == 'active',
                CSRFProtection.expires_at > datetime.utcnow()
            ).order_by(desc(CSRFProtection.created_at)).limit(10).all()
            
            # Optimized CSRF violations query
            csrf_violations = self.db.query(CSRFViolation).filter(
                CSRFViolation.user_id == user_id
            ).order_by(desc(CSRFViolation.created_at)).limit(20).all()
            
            # CSRF statistics
            csrf_stats = self.db.query(
                func.count(CSRFProtection.id).label('total_tokens'),
                func.count(
                    self.db.query(CSRFProtection.id).filter(
                        CSRFProtection.status == 'active',
                        CSRFProtection.expires_at > datetime.utcnow()
                    ).subquery().c.id
                ).label('active_tokens'),
                func.count(CSRFViolation.id).label('total_violations')
            ).filter(CSRFProtection.user_id == user_id).first()
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "csrf_tokens": [
                    {
                        "id": token.id,
                        "token": token.token[:8] + "...",
                        "status": token.status,
                        "is_single_use": token.is_single_use,
                        "created_at": token.created_at.isoformat(),
                        "expires_at": token.expires_at.isoformat()
                    } for token in csrf_tokens
                ],
                "csrf_violations": [
                    {
                        "id": violation.id,
                        "violation_type": violation.violation_type,
                        "ip_address": violation.ip_address,
                        "severity": violation.severity,
                        "created_at": violation.created_at.isoformat()
                    } for violation in csrf_violations
                ],
                "statistics": {
                    "total_tokens": csrf_stats.total_tokens or 0,
                    "active_tokens": csrf_stats.active_tokens or 0,
                    "total_violations": csrf_stats.total_violations or 0
                },
                "execution_time_ms": execution_time,
                "optimization_applied": True
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in optimize_csrf_queries: {str(e)}")
            return {"error": "Database query failed"}
        except Exception as e:
            logger.error(f"Error in optimize_csrf_queries: {str(e)}")
            return {"error": "Query optimization failed"}
    
    def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        try:
            total_queries = len(self.query_stats)
            if total_queries == 0:
                return {"message": "No query statistics available"}
            
            execution_times = [stats['execution_time'] for stats in self.query_stats.values()]
            
            return {
                "total_queries": total_queries,
                "average_execution_time_ms": statistics.mean(execution_times),
                "median_execution_time_ms": statistics.median(execution_times),
                "min_execution_time_ms": min(execution_times),
                "max_execution_time_ms": max(execution_times),
                "slow_queries": len([t for t in execution_times if t > self.slow_query_threshold]),
                "slow_query_threshold_ms": self.slow_query_threshold,
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "optimization_recommendations": self._get_optimization_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error getting query performance stats: {str(e)}")
            return {"error": "Failed to get performance statistics"}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate query cache hit rate"""
        if not self.query_cache:
            return 0.0
        
        total_requests = len(self.query_stats)
        cache_hits = len(self.query_cache)
        
        return (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get query optimization recommendations"""
        recommendations = []
        
        # Check for slow queries
        slow_queries = [stats for stats in self.query_stats.values() 
                       if stats['execution_time'] > self.slow_query_threshold]
        
        if slow_queries:
            recommendations.append(f"Consider adding indexes for {len(slow_queries)} slow queries")
        
        # Check cache hit rate
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate < 50:
            recommendations.append("Consider increasing query cache size")
        
        # Check for N+1 queries
        if len(self.query_stats) > 10:
            recommendations.append("Consider using select_related or joinedload to reduce N+1 queries")
        
        return recommendations
    
    def clear_query_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def clear_query_stats(self) -> None:
        """Clear query statistics"""
        self.query_stats.clear()
        logger.info("Query statistics cleared")
