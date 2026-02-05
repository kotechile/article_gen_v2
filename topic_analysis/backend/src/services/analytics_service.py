"""
Analytics Service
Provides analytics, reporting, and insights for the TrendTap platform
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache

logger = structlog.get_logger()

class AnalyticsService:
    """Service for analytics, reporting, and insights"""
    
    def __init__(self):
        self.db = get_supabase_db()
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def get_user_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a user
        
        Args:
            user_id: User ID
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            
        Returns:
            Dict containing user analytics
        """
        try:
            logger.info("Getting user analytics", 
                       user_id=user_id,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get analytics data
            analytics_data = await self._gather_analytics_data(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate insights
            insights = await self._calculate_insights(analytics_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(analytics_data, insights)
            
            logger.info("User analytics calculated", 
                       user_id=user_id,
                       insights_count=len(insights),
                       recommendations_count=len(recommendations))
            
            return {
                "success": True,
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "analytics": analytics_data,
                "insights": insights,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get user analytics", error=str(e))
            raise
    
    async def get_platform_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get platform-wide analytics (admin only)
        
        Args:
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            
        Returns:
            Dict containing platform analytics
        """
        try:
            logger.info("Getting platform analytics", 
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get platform data
            platform_data = await self._gather_platform_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate platform insights
            insights = await self._calculate_platform_insights(platform_data)
            
            logger.info("Platform analytics calculated", 
                       total_users=platform_data.get("total_users", 0),
                       total_analyses=platform_data.get("total_analyses", 0))
            
            return {
                "success": True,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "platform_data": platform_data,
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get platform analytics", error=str(e))
            raise
    
    async def get_content_performance_analytics(
        self,
        user_id: str,
        content_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get content performance analytics
        
        Args:
            user_id: User ID
            content_ids: Optional list of content IDs to analyze
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            
        Returns:
            Dict containing content performance analytics
        """
        try:
            logger.info("Getting content performance analytics", 
                       user_id=user_id,
                       content_count=len(content_ids) if content_ids else None,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Get content performance data
            performance_data = await self._gather_content_performance_data(
                user_id=user_id,
                content_ids=content_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics(performance_data)
            
            # Generate content insights
            insights = await self._generate_content_insights(performance_data, metrics)
            
            logger.info("Content performance analytics calculated", 
                       content_count=len(performance_data.get("content_items", [])),
                       metrics_count=len(metrics))
            
            return {
                "success": True,
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                },
                "performance_data": performance_data,
                "metrics": metrics,
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get content performance analytics", error=str(e))
            raise
    
    async def get_trend_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get trend analysis analytics
        
        Args:
            user_id: User ID
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            
        Returns:
            Dict containing trend analytics
        """
        try:
            logger.info("Getting trend analytics", 
                       user_id=user_id,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Get trend data
            trend_data = await self._gather_trend_data(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate trend insights
            insights = await self._calculate_trend_insights(trend_data)
            
            # Generate trend predictions
            predictions = await self._generate_trend_predictions(trend_data, insights)
            
            logger.info("Trend analytics calculated", 
                       trend_analyses=len(trend_data.get("trend_analyses", [])),
                       insights_count=len(insights))
            
            return {
                "success": True,
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                },
                "trend_data": trend_data,
                "insights": insights,
                "predictions": predictions,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get trend analytics", error=str(e))
            raise
    
    async def export_analytics_report(
        self,
        user_id: str,
        report_type: str,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Export analytics report in specified format
        
        Args:
            user_id: User ID
            report_type: Type of report (user, content, trend, platform)
            format: Export format (json, csv, pdf)
            start_date: Optional start date for report
            end_date: Optional end date for report
            
        Returns:
            Dict containing export results
        """
        try:
            logger.info("Exporting analytics report", 
                       user_id=user_id,
                       report_type=report_type,
                       format=format,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Get analytics data based on report type
            if report_type == "user":
                analytics_data = await self.get_user_analytics(
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date
                )
            elif report_type == "content":
                analytics_data = await self.get_content_performance_analytics(
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date
                )
            elif report_type == "trend":
                analytics_data = await self.get_trend_analytics(
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date
                )
            elif report_type == "platform":
                analytics_data = await self.get_platform_analytics(
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                raise ValueError(f"Invalid report type: {report_type}")
            
            # Generate export file
            export_result = await self._generate_export_file(
                analytics_data=analytics_data,
                format=format,
                report_type=report_type
            )
            
            logger.info("Analytics report exported", 
                       report_type=report_type,
                       format=format,
                       file_size=export_result.get("file_size", 0))
            
            return {
                "success": True,
                "report_type": report_type,
                "format": format,
                "export_result": export_result,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export analytics report", error=str(e))
            raise
    
    async def _gather_analytics_data(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Gather analytics data for user"""
        try:
            # Get user activity data
            activity_data = await self.db.get_user_activity(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get content performance data
            content_data = await self.db.get_content_performance(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get trend analysis data
            trend_data = await self.db.get_trend_analyses(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "activity_data": activity_data,
                "content_data": content_data,
                "trend_data": trend_data,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to gather analytics data", error=str(e))
            raise
    
    async def _gather_platform_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Gather platform-wide data"""
        try:
            # Get platform statistics
            platform_stats = await self.db.get_platform_statistics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Get user growth data
            user_growth = await self.db.get_user_growth(
                start_date=start_date,
                end_date=end_date
            )
            
            # Get content creation data
            content_creation = await self.db.get_content_creation_stats(
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "platform_stats": platform_stats,
                "user_growth": user_growth,
                "content_creation": content_creation,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to gather platform data", error=str(e))
            raise
    
    async def _gather_content_performance_data(
        self,
        user_id: str,
        content_ids: Optional[List[str]],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, Any]:
        """Gather content performance data"""
        try:
            # Get content performance metrics
            performance_metrics = await self.db.get_content_performance_metrics(
                user_id=user_id,
                content_ids=content_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get content engagement data
            engagement_data = await self.db.get_content_engagement(
                user_id=user_id,
                content_ids=content_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "performance_metrics": performance_metrics,
                "engagement_data": engagement_data,
                "content_ids": content_ids,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            }
            
        except Exception as e:
            logger.error("Failed to gather content performance data", error=str(e))
            raise
    
    async def _gather_trend_data(
        self,
        user_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, Any]:
        """Gather trend analysis data"""
        try:
            # Get trend analyses
            trend_analyses = await self.db.get_trend_analyses(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get trend accuracy data
            accuracy_data = await self.db.get_trend_accuracy(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "trend_analyses": trend_analyses,
                "accuracy_data": accuracy_data,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            }
            
        except Exception as e:
            logger.error("Failed to gather trend data", error=str(e))
            raise
    
    async def _calculate_insights(self, analytics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate insights from analytics data"""
        insights = []
        
        # Activity insights
        activity_data = analytics_data.get("activity_data", {})
        if activity_data:
            insights.append({
                "type": "activity",
                "title": "Activity Summary",
                "description": f"User has been active for {activity_data.get('active_days', 0)} days",
                "value": activity_data.get('active_days', 0),
                "trend": "increasing" if activity_data.get('trend', 0) > 0 else "stable"
            })
        
        # Content insights
        content_data = analytics_data.get("content_data", {})
        if content_data:
            insights.append({
                "type": "content",
                "title": "Content Performance",
                "description": f"Generated {content_data.get('total_content', 0)} content pieces",
                "value": content_data.get('total_content', 0),
                "trend": "increasing" if content_data.get('trend', 0) > 0 else "stable"
            })
        
        return insights
    
    async def _calculate_platform_insights(self, platform_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate platform insights"""
        insights = []
        
        # User growth insights
        user_growth = platform_data.get("user_growth", {})
        if user_growth:
            insights.append({
                "type": "growth",
                "title": "User Growth",
                "description": f"Platform has {user_growth.get('total_users', 0)} total users",
                "value": user_growth.get('total_users', 0),
                "trend": "increasing" if user_growth.get('growth_rate', 0) > 0 else "stable"
            })
        
        return insights
    
    async def _calculate_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        metrics = {}
        
        # Calculate average performance
        performance_metrics = performance_data.get("performance_metrics", [])
        if performance_metrics:
            metrics["average_performance"] = sum(
                p.get("score", 0) for p in performance_metrics
            ) / len(performance_metrics)
        
        return metrics
    
    async def _generate_content_insights(
        self,
        performance_data: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate content insights"""
        insights = []
        
        # Performance insights
        avg_performance = metrics.get("average_performance", 0)
        if avg_performance > 0.8:
            insights.append({
                "type": "performance",
                "title": "High Performance",
                "description": "Your content is performing above average",
                "recommendation": "Continue with current strategy"
            })
        elif avg_performance < 0.5:
            insights.append({
                "type": "performance",
                "title": "Low Performance",
                "description": "Your content could benefit from optimization",
                "recommendation": "Review content strategy and keywords"
            })
        
        return insights
    
    async def _calculate_trend_insights(self, trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate trend insights"""
        insights = []
        
        # Trend accuracy insights
        accuracy_data = trend_data.get("accuracy_data", {})
        if accuracy_data:
            accuracy = accuracy_data.get("accuracy", 0)
            insights.append({
                "type": "accuracy",
                "title": "Trend Accuracy",
                "description": f"Trend predictions are {accuracy:.1%} accurate",
                "value": accuracy,
                "trend": "improving" if accuracy > 0.7 else "needs_improvement"
            })
        
        return insights
    
    async def _generate_trend_predictions(
        self,
        trend_data: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate trend predictions"""
        predictions = []
        
        # Generate predictions based on historical data
        trend_analyses = trend_data.get("trend_analyses", [])
        if trend_analyses:
            predictions.append({
                "type": "trend",
                "title": "Upcoming Trends",
                "description": "Based on historical data, expect increased activity in trending topics",
                "confidence": 0.75,
                "timeframe": "30 days"
            })
        
        return predictions
    
    async def _generate_recommendations(
        self,
        analytics_data: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        # Activity-based recommendations
        activity_data = analytics_data.get("activity_data", {})
        if activity_data.get("active_days", 0) < 7:
            recommendations.append({
                "type": "engagement",
                "title": "Increase Activity",
                "description": "Consider using the platform more regularly for better results",
                "priority": "medium"
            })
        
        # Content-based recommendations
        content_data = analytics_data.get("content_data", {})
        if content_data.get("total_content", 0) < 5:
            recommendations.append({
                "type": "content",
                "title": "Create More Content",
                "description": "Generate more content ideas to improve your content strategy",
                "priority": "high"
            })
        
        return recommendations
    
    async def _generate_export_file(
        self,
        analytics_data: Dict[str, Any],
        format: str,
        report_type: str
    ) -> Dict[str, Any]:
        """Generate export file in specified format"""
        try:
            if format == "json":
                return {
                    "file_type": "json",
                    "file_size": len(json.dumps(analytics_data)),
                    "download_url": f"/exports/{report_type}_report.json"
                }
            elif format == "csv":
                # Convert to CSV format
                return {
                    "file_type": "csv",
                    "file_size": 0,  # Would be calculated
                    "download_url": f"/exports/{report_type}_report.csv"
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error("Failed to generate export file", error=str(e))
            raise

