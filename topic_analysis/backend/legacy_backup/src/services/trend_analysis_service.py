"""
Trend Analysis Service
Handles trend analysis operations and data processing
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ..models.trend_analysis import TrendAnalysis, TrendAnalysisStatus, TrendAnalysisSource
from ..services.google_trends_service import GoogleTrendsService
from ..services.csv_upload_service import CSVUploadService
from ..core.redis import cache

logger = structlog.get_logger()

class TrendAnalysisService:
    """Service for managing trend analysis operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.google_trends_service = GoogleTrendsService(db)
        self.csv_upload_service = CSVUploadService(db)
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def create_trend_analysis(
        self,
        user_id: str,
        workflow_session_id: str,
        analysis_name: str,
        keywords: List[str],
        timeframe: str = "12m",
        geo: str = "US",
        category: Optional[int] = None,
        description: Optional[str] = None,
        topic_decomposition_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new trend analysis"""
        try:
            logger.info("Creating trend analysis", 
                       user_id=user_id, 
                       analysis_name=analysis_name,
                       keywords=keywords)
            
            analysis_id = str(uuid.uuid4())
            
            trend_analysis = TrendAnalysis(
                id=analysis_id,
                user_id=user_id,
                workflow_session_id=workflow_session_id,
                topic_decomposition_id=topic_decomposition_id,
                analysis_name=analysis_name,
                description=description,
                keywords=keywords,
                timeframe=timeframe,
                geo=geo,
                category=category,
                status=TrendAnalysisStatus.PENDING
            )
            
            self.db.add(trend_analysis)
            self.db.commit()
            self.db.refresh(trend_analysis)
            
            logger.info("Trend analysis created successfully", analysis_id=analysis_id)
            return trend_analysis.to_dict()
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Integrity error creating trend analysis", error=str(e))
            raise ValueError("Failed to create trend analysis due to data constraints")
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to create trend analysis", error=str(e))
            raise
    
    async def start_trend_analysis(
        self,
        analysis_id: str,
        user_id: str,
        source: TrendAnalysisSource = TrendAnalysisSource.GOOGLE_TRENDS
    ) -> Dict[str, Any]:
        """Start trend analysis processing"""
        try:
            logger.info("Starting trend analysis", analysis_id=analysis_id, source=source.value)
            
            # Get analysis record
            trend_analysis = self.db.query(TrendAnalysis).filter(
                TrendAnalysis.id == analysis_id,
                TrendAnalysis.user_id == user_id
            ).first()
            
            if not trend_analysis:
                raise ValueError("Trend analysis not found")
            
            if trend_analysis.status != TrendAnalysisStatus.PENDING:
                raise ValueError(f"Analysis is not in pending status: {trend_analysis.status.value}")
            
            # Update status to in progress
            trend_analysis.status = TrendAnalysisStatus.IN_PROGRESS
            trend_analysis.source = source
            trend_analysis.updated_at = datetime.utcnow()
            self.db.commit()
            
            # Start processing based on source
            start_time = datetime.utcnow()
            
            try:
                if source == TrendAnalysisSource.GOOGLE_TRENDS:
                    trend_data = await self._process_google_trends_analysis(trend_analysis)
                elif source == TrendAnalysisSource.CSV_UPLOAD:
                    trend_data = await self._process_csv_upload_analysis(trend_analysis)
                else:
                    raise ValueError(f"Unsupported source: {source.value}")
                
                # Process analysis results
                analysis_results = await self._process_analysis_results(trend_data, trend_analysis)
                
                # Generate insights
                insights = await self._generate_insights(analysis_results, trend_analysis)
                
                # Update analysis with results
                end_time = datetime.utcnow()
                processing_time = int((end_time - start_time).total_seconds() * 1000)
                
                trend_analysis.trend_data = trend_data
                trend_analysis.analysis_results = analysis_results
                trend_analysis.insights = insights
                trend_analysis.status = TrendAnalysisStatus.COMPLETED
                trend_analysis.processing_time_ms = processing_time
                trend_analysis.completed_at = end_time
                trend_analysis.updated_at = end_time
                
                self.db.commit()
                
                logger.info("Trend analysis completed successfully", 
                           analysis_id=analysis_id,
                           processing_time_ms=processing_time)
                
                return trend_analysis.to_dict()
                
            except Exception as e:
                # Mark as failed
                trend_analysis.status = TrendAnalysisStatus.FAILED
                trend_analysis.error_message = str(e)
                trend_analysis.updated_at = datetime.utcnow()
                self.db.commit()
                
                logger.error("Trend analysis failed", analysis_id=analysis_id, error=str(e))
                raise
            
        except Exception as e:
            logger.error("Failed to start trend analysis", analysis_id=analysis_id, error=str(e))
            raise
    
    async def _process_google_trends_analysis(self, trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Process Google Trends analysis"""
        try:
            logger.info("Processing Google Trends analysis", analysis_id=trend_analysis.id)
            
            trend_data = await self.google_trends_service.get_trend_data(
                keywords=trend_analysis.keywords,
                user_id=trend_analysis.user_id,
                timeframe=trend_analysis.timeframe,
                geo=trend_analysis.geo,
                category=trend_analysis.category
            )
            
            # Update API calls count
            trend_analysis.api_calls_made = trend_analysis.api_calls_made + 1
            
            return trend_data
            
        except Exception as e:
            logger.error("Google Trends analysis failed", analysis_id=trend_analysis.id, error=str(e))
            raise
    
    async def _process_csv_upload_analysis(self, trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Process CSV upload analysis"""
        try:
            logger.info("Processing CSV upload analysis", analysis_id=trend_analysis.id)
            
            # Get cached CSV data
            cached_data = await self.csv_upload_service.get_cached_trend_data(
                user_id=trend_analysis.user_id,
                workflow_session_id=trend_analysis.workflow_session_id
            )
            
            if not cached_data:
                raise ValueError("No CSV data found for analysis")
            
            return cached_data.get("trend_data", {})
            
        except Exception as e:
            logger.error("CSV upload analysis failed", analysis_id=trend_analysis.id, error=str(e))
            raise
    
    async def _process_analysis_results(self, trend_data: Dict[str, Any], trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Process raw trend data into analysis results"""
        try:
            logger.info("Processing analysis results", analysis_id=trend_analysis.id)
            
            trends = trend_data.get("trends", [])
            if not trends:
                return {"error": "No trend data available"}
            
            # Calculate summary statistics
            total_keywords = len(trends)
            total_volume = sum(trend.get("search_volume", 0) for trend in trends)
            avg_volume = total_volume / total_keywords if total_keywords > 0 else 0
            
            # Find top performing keywords
            top_keywords = sorted(trends, key=lambda x: x.get("search_volume", 0), reverse=True)[:10]
            
            # Calculate growth metrics
            growing_keywords = [t for t in trends if t.get("growth_rate", 0) > 0]
            declining_keywords = [t for t in trends if t.get("growth_rate", 0) < 0]
            
            # Analyze competition levels
            competition_levels = {}
            for trend in trends:
                comp = trend.get("competition", "unknown")
                competition_levels[comp] = competition_levels.get(comp, 0) + 1
            
            # Generate time series analysis
            time_series_analysis = self._analyze_time_series(trends)
            
            analysis_results = {
                "summary": {
                    "total_keywords": total_keywords,
                    "total_volume": total_volume,
                    "average_volume": avg_volume,
                    "growing_keywords": len(growing_keywords),
                    "declining_keywords": len(declining_keywords),
                    "competition_levels": competition_levels
                },
                "top_keywords": top_keywords,
                "growing_keywords": growing_keywords[:5],
                "declining_keywords": declining_keywords[:5],
                "time_series_analysis": time_series_analysis,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error("Failed to process analysis results", analysis_id=trend_analysis.id, error=str(e))
            return {"error": f"Failed to process results: {str(e)}"}
    
    async def _generate_insights(self, analysis_results: Dict[str, Any], trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Generate insights from analysis results"""
        try:
            logger.info("Generating insights", analysis_id=trend_analysis.id)
            
            insights = []
            top_trending = []
            growth_opportunities = []
            seasonal_patterns = []
            
            # Analyze top keywords
            top_keywords = analysis_results.get("top_keywords", [])
            for keyword in top_keywords[:5]:
                top_trending.append({
                    "keyword": keyword.get("keyword", ""),
                    "search_volume": keyword.get("search_volume", 0),
                    "trend_score": keyword.get("trend_score", 0),
                    "reason": "High search volume"
                })
            
            # Analyze growth opportunities
            growing_keywords = analysis_results.get("growing_keywords", [])
            for keyword in growing_keywords:
                growth_opportunities.append({
                    "keyword": keyword.get("keyword", ""),
                    "growth_rate": keyword.get("growth_rate", 0),
                    "search_volume": keyword.get("search_volume", 0),
                    "reason": f"Growing at {keyword.get('growth_rate', 0):.1f}%"
                })
            
            # Analyze seasonal patterns
            time_series = analysis_results.get("time_series_analysis", {})
            if time_series.get("seasonal_keywords"):
                for keyword in time_series["seasonal_keywords"]:
                    seasonal_patterns.append({
                        "keyword": keyword.get("keyword", ""),
                        "pattern": keyword.get("pattern", ""),
                        "peak_months": keyword.get("peak_months", []),
                        "reason": "Shows seasonal patterns"
                    })
            
            # Generate overall insights
            summary = analysis_results.get("summary", {})
            if summary.get("growing_keywords", 0) > summary.get("declining_keywords", 0):
                insights.append("Overall positive growth trend detected")
            elif summary.get("declining_keywords", 0) > summary.get("growing_keywords", 0):
                insights.append("Overall declining trend detected")
            else:
                insights.append("Stable trend pattern detected")
            
            if summary.get("average_volume", 0) > 1000:
                insights.append("High-volume keywords present - good for broad reach")
            elif summary.get("average_volume", 0) > 100:
                insights.append("Medium-volume keywords - balanced approach recommended")
            else:
                insights.append("Low-volume keywords - consider niche targeting")
            
            return {
                "insights": insights,
                "top_trending": top_trending,
                "growth_opportunities": growth_opportunities,
                "seasonal_patterns": seasonal_patterns,
                "competition_level": self._analyze_competition_level(summary.get("competition_levels", {})),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to generate insights", analysis_id=trend_analysis.id, error=str(e))
            return {"error": f"Failed to generate insights: {str(e)}"}
    
    def _analyze_time_series(self, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time series patterns"""
        try:
            seasonal_keywords = []
            stable_keywords = []
            
            for trend in trends:
                time_series = trend.get("time_series", [])
                if len(time_series) < 12:  # Need at least 12 data points for seasonal analysis
                    continue
                
                # Simple seasonal analysis
                values = [point.get("value", 0) for point in time_series]
                if len(values) >= 12:
                    # Calculate variance
                    mean_val = sum(values) / len(values)
                    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                    
                    if variance > mean_val * 0.5:  # High variance indicates seasonality
                        seasonal_keywords.append({
                            "keyword": trend.get("keyword", ""),
                            "pattern": "seasonal",
                            "peak_months": self._find_peak_months(values),
                            "variance": variance
                        })
                    else:
                        stable_keywords.append({
                            "keyword": trend.get("keyword", ""),
                            "pattern": "stable",
                            "variance": variance
                        })
            
            return {
                "seasonal_keywords": seasonal_keywords,
                "stable_keywords": stable_keywords,
                "total_analyzed": len(seasonal_keywords) + len(stable_keywords)
            }
            
        except Exception as e:
            logger.error("Failed to analyze time series", error=str(e))
            return {"error": f"Failed to analyze time series: {str(e)}"}
    
    def _find_peak_months(self, values: List[float]) -> List[int]:
        """Find peak months from time series values"""
        try:
            if len(values) < 12:
                return []
            
            # Find months with values above average
            mean_val = sum(values) / len(values)
            peak_months = [i for i, v in enumerate(values) if v > mean_val * 1.2]
            return peak_months[:3]  # Return top 3 peak months
        except:
            return []
    
    def _analyze_competition_level(self, competition_levels: Dict[str, int]) -> str:
        """Analyze overall competition level"""
        try:
            if not competition_levels:
                return "unknown"
            
            total = sum(competition_levels.values())
            if total == 0:
                return "unknown"
            
            # Calculate weighted competition score
            high_comp = competition_levels.get("high", 0) + competition_levels.get("very_high", 0)
            medium_comp = competition_levels.get("medium", 0)
            low_comp = competition_levels.get("low", 0) + competition_levels.get("very_low", 0)
            
            if high_comp / total > 0.5:
                return "high"
            elif medium_comp / total > 0.5:
                return "medium"
            elif low_comp / total > 0.5:
                return "low"
            else:
                return "mixed"
        except:
            return "unknown"
    
    async def get_trend_analysis(self, analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get trend analysis by ID"""
        try:
            trend_analysis = self.db.query(TrendAnalysis).filter(
                TrendAnalysis.id == analysis_id,
                TrendAnalysis.user_id == user_id
            ).first()
            
            if not trend_analysis:
                return None
            
            return trend_analysis.to_dict()
            
        except Exception as e:
            logger.error("Failed to get trend analysis", analysis_id=analysis_id, error=str(e))
            raise
    
    async def list_trend_analyses(self, user_id: str, workflow_session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List trend analyses for user"""
        try:
            query = self.db.query(TrendAnalysis).filter(TrendAnalysis.user_id == user_id)
            
            if workflow_session_id:
                query = query.filter(TrendAnalysis.workflow_session_id == workflow_session_id)
            
            trend_analyses = query.order_by(TrendAnalysis.created_at.desc()).all()
            
            return [ta.to_summary_dict() for ta in trend_analyses]
            
        except Exception as e:
            logger.error("Failed to list trend analyses", user_id=user_id, error=str(e))
            raise
    
    async def update_trend_analysis(
        self,
        analysis_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update trend analysis"""
        try:
            trend_analysis = self.db.query(TrendAnalysis).filter(
                TrendAnalysis.id == analysis_id,
                TrendAnalysis.user_id == user_id
            ).first()
            
            if not trend_analysis:
                return None
            
            # Update allowed fields
            allowed_fields = ["analysis_name", "description", "timeframe", "geo", "category"]
            for field, value in updates.items():
                if field in allowed_fields and hasattr(trend_analysis, field):
                    setattr(trend_analysis, field, value)
            
            trend_analysis.updated_at = datetime.utcnow()
            self.db.commit()
            
            return trend_analysis.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to update trend analysis", analysis_id=analysis_id, error=str(e))
            raise
    
    async def delete_trend_analysis(self, analysis_id: str, user_id: str) -> bool:
        """Delete trend analysis"""
        try:
            trend_analysis = self.db.query(TrendAnalysis).filter(
                TrendAnalysis.id == analysis_id,
                TrendAnalysis.user_id == user_id
            ).first()
            
            if not trend_analysis:
                return False
            
            self.db.delete(trend_analysis)
            self.db.commit()
            
            logger.info("Trend analysis deleted successfully", analysis_id=analysis_id)
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete trend analysis", analysis_id=analysis_id, error=str(e))
            raise