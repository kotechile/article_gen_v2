"""
Trend Analysis Background Tasks
"""

from celery import current_task
from celery.exceptions import Retry
import structlog

from ..core.celery_app import celery_app
from ..core.database import get_db_session
from ..core.redis import cache_manager
from ..integrations.google_trends import get_trend_data, get_interest_over_time

logger = structlog.get_logger()

@celery_app.task(bind=True, max_retries=3)
def analyze_trends_task(self, keywords: list, user_id: str, analysis_id: str):
    """Analyze trends for keywords in background"""
    try:
        # Update task status
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Analyzing trends", "progress": 0}
        )
        
        results = {}
        
        for i, keyword in enumerate(keywords):
            try:
                # Get trend data
                trend_data = get_trend_data(keyword, geo="US", timeframe="today 12-m")
                results[keyword] = trend_data
                
                # Update progress
                progress = int((i + 1) / len(keywords) * 100)
                current_task.update_state(
                    state="PROGRESS",
                    meta={"status": f"Analyzing keyword: {keyword}", "progress": progress}
                )
                
                # Cache trend data
                cache_manager.cache_trend_data(keyword, "US", trend_data, expire=3600)
                
            except Exception as e:
                logger.error(
                    "Failed to analyze trend for keyword",
                    keyword=keyword,
                    error=str(e)
                )
                results[keyword] = {"error": str(e)}
        
        # Save results to database
        with get_db_session() as db:
            from ..models.trend_analysis import TrendAnalysis
            analysis = db.get_TrendAnalysis_by_id(
                TrendAnalysis.id == analysis_id
            )
            
            if analysis:
                analysis.status = "completed"
                analysis.results = results
                db.commit()
        
        # Update final status
        current_task.update_state(
            state="SUCCESS",
            meta={"status": "Completed", "progress": 100, "keywords_analyzed": len(keywords)}
        )
        
        logger.info(
            "Trend analysis completed",
            user_id=user_id,
            analysis_id=analysis_id,
            keywords_count=len(keywords)
        )
        
        return {"status": "success", "keywords_analyzed": len(keywords)}
        
    except Exception as e:
        logger.error(
            "Trend analysis failed",
            user_id=user_id,
            analysis_id=analysis_id,
            error=str(e)
        )
        
        # Retry with exponential backoff
        try:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        except Retry:
            raise
        except Exception:
            # Update task status to failed
            current_task.update_state(
                state="FAILURE",
                meta={"status": "Failed", "error": str(e)}
            )
            raise

@celery_app.task(bind=True)
def update_trend_data(self):
    """Update trend data cache for popular keywords"""
    try:
        logger.info("Starting trend data update")
        
        # Get popular keywords from database
        popular_keywords = [
            "artificial intelligence",
            "cryptocurrency",
            "sustainable living",
            "remote work",
            "mental health",
            "climate change",
            "electric vehicles",
            "renewable energy",
            "digital marketing",
            "e-commerce"
        ]
        
        for keyword in popular_keywords:
            try:
                # Get trend data
                trend_data = get_trend_data(keyword, geo="US", timeframe="today 12-m")
                
                # Cache trend data
                cache_manager.cache_trend_data(keyword, "US", trend_data, expire=86400)  # 24 hours
                
                logger.info(
                    "Updated trend data for keyword",
                    keyword=keyword
                )
                
            except Exception as e:
                logger.error(
                    "Failed to update trend data for keyword",
                    keyword=keyword,
                    error=str(e)
                )
        
        logger.info("Trend data update completed")
        return {"status": "success", "updated_keywords": len(popular_keywords)}
        
    except Exception as e:
        logger.error("Trend data update failed", error=str(e))
        raise

@celery_app.task(bind=True)
def compare_trends_task(self, keywords: list, user_id: str, comparison_id: str):
    """Compare trends between multiple keywords"""
    try:
        logger.info(
            "Starting trend comparison",
            keywords=keywords,
            user_id=user_id,
            comparison_id=comparison_id
        )
        
        # Get interest over time for all keywords
        interest_data = get_interest_over_time(keywords, geo="US", timeframe="today 12-m")
        
        # Perform comparison analysis
        comparison_results = {
            "keywords": keywords,
            "interest_data": interest_data,
            "comparison": {},
            "insights": []
        }
        
        # Calculate comparison metrics
        if "comparison" in interest_data:
            for keyword, metrics in interest_data["comparison"].items():
                comparison_results["comparison"][keyword] = {
                    "average_interest": metrics.get("average", 0),
                    "peak_interest": metrics.get("peak", 0),
                    "total_interest": metrics.get("total", 0)
                }
        
        # Generate insights
        if comparison_results["comparison"]:
            # Find highest performing keyword
            highest_keyword = max(
                comparison_results["comparison"].items(),
                key=lambda x: x[1]["average_interest"]
            )[0]
            
            comparison_results["insights"].append({
                "type": "highest_performing",
                "keyword": highest_keyword,
                "value": comparison_results["comparison"][highest_keyword]["average_interest"]
            })
        
        # Save results to database
        with get_db_session() as db:
            from ..models.trend_analysis import TrendAnalysis
            analysis = db.get_TrendAnalysis_by_id(
                TrendAnalysis.id == comparison_id
            )
            
            if analysis:
                analysis.status = "completed"
                analysis.results = comparison_results
                db.commit()
        
        logger.info(
            "Trend comparison completed",
            user_id=user_id,
            comparison_id=comparison_id
        )
        
        return {"status": "success", "comparison_results": comparison_results}
        
    except Exception as e:
        logger.error(
            "Trend comparison failed",
            user_id=user_id,
            comparison_id=comparison_id,
            error=str(e)
        )
        raise

@celery_app.task(bind=True)
def generate_trend_insights_task(self, trend_data: dict, user_id: str):
    """Generate AI-powered insights from trend data"""
    try:
        logger.info(
            "Starting trend insights generation",
            user_id=user_id
        )
        
        # This would typically involve:
        # 1. Analyzing trend patterns
        # 2. Using AI to generate insights
        # 3. Creating actionable recommendations
        
        # Mock implementation
        insights = {
            "trend_direction": "rising",
            "confidence_score": 0.85,
            "key_insights": [
                "Interest in this topic has increased 25% over the past 3 months",
                "Peak interest occurs during weekday mornings",
                "Related topics show similar growth patterns"
            ],
            "recommendations": [
                "Consider creating content around this topic",
                "Monitor competitor activity in this space",
                "Plan content calendar around peak interest times"
            ],
            "risk_factors": [
                "Interest may be seasonal",
                "Competition is increasing"
            ]
        }
        
        logger.info(
            "Trend insights generation completed",
            user_id=user_id
        )
        
        return {"status": "success", "insights": insights}
        
    except Exception as e:
        logger.error(
            "Trend insights generation failed",
            user_id=user_id,
            error=str(e)
        )
        raise
