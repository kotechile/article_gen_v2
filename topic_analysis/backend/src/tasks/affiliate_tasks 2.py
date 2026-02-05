"""
Affiliate Research Background Tasks
"""

from celery import current_task
from celery.exceptions import Retry
import structlog

from ..core.celery_app import celery_app
from ..core.database import get_db_session
from ..core.redis import cache_manager
from ..integrations.affiliate_networks import search_affiliate_programs

logger = structlog.get_logger()

@celery_app.task(bind=True, max_retries=3)
def search_affiliate_programs_task(self, niche: str, user_id: str, research_id: str):
    """Search affiliate programs in background"""
    try:
        # Update task status
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Searching affiliate programs", "progress": 0}
        )
        
        # Search affiliate programs
        results = search_affiliate_programs(niche, limit_per_network=10)
        
        # Update progress
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Processing results", "progress": 50}
        )
        
        # Save results to database
        with get_db_session() as db:
            # Update affiliate research record
            from ..models.affiliate_research import AffiliateResearch
            research = db.get_AffiliateResearch_by_id(
                AffiliateResearch.id == research_id
            )
            
            if research:
                research.status = "completed"
                research.results = results
                db.commit()
        
        # Cache results
        cache_manager.cache_affiliate_programs(niche, results, expire=3600)
        
        # Update final status
        current_task.update_state(
            state="SUCCESS",
            meta={"status": "Completed", "progress": 100, "results": len(results)}
        )
        
        logger.info(
            "Affiliate programs search completed",
            niche=niche,
            user_id=user_id,
            research_id=research_id,
            results_count=len(results)
        )
        
        return {"status": "success", "results_count": len(results)}
        
    except Exception as e:
        logger.error(
            "Affiliate programs search failed",
            niche=niche,
            user_id=user_id,
            research_id=research_id,
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
def update_affiliate_programs(self):
    """Update affiliate programs cache"""
    try:
        logger.info("Starting affiliate programs update")
        
        # Get popular niches
        popular_niches = [
            "health and fitness",
            "technology",
            "finance",
            "travel",
            "food and cooking",
            "beauty and fashion",
            "home and garden",
            "education",
            "business",
            "entertainment"
        ]
        
        for niche in popular_niches:
            try:
                # Search and cache programs
                results = search_affiliate_programs(niche, limit_per_network=5)
                cache_manager.cache_affiliate_programs(niche, results, expire=86400)  # 24 hours
                
                logger.info(
                    "Updated affiliate programs for niche",
                    niche=niche,
                    count=len(results)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to update affiliate programs for niche",
                    niche=niche,
                    error=str(e)
                )
        
        logger.info("Affiliate programs update completed")
        return {"status": "success", "updated_niches": len(popular_niches)}
        
    except Exception as e:
        logger.error("Affiliate programs update failed", error=str(e))
        raise

@celery_app.task(bind=True)
def analyze_affiliate_program_performance(self, program_id: str, user_id: str):
    """Analyze affiliate program performance"""
    try:
        logger.info(
            "Starting affiliate program performance analysis",
            program_id=program_id,
            user_id=user_id
        )
        
        # This would typically involve:
        # 1. Fetching program data
        # 2. Analyzing performance metrics
        # 3. Updating database with analysis results
        
        # Mock implementation
        analysis_results = {
            "program_id": program_id,
            "performance_score": 85,
            "recommendation": "high_performing",
            "metrics": {
                "conversion_rate": 0.15,
                "average_commission": 25.50,
                "payout_frequency": "monthly"
            }
        }
        
        # Save analysis results
        with get_db_session() as db:
            # Update affiliate research record with analysis
            from ..models.affiliate_research import AffiliateResearch
            research = db.get_AffiliateResearch_by_id(
                AffiliateResearch.id == program_id
            )
            
            if research:
                research.analysis_results = analysis_results
                db.commit()
        
        logger.info(
            "Affiliate program performance analysis completed",
            program_id=program_id,
            user_id=user_id
        )
        
        return {"status": "success", "analysis": analysis_results}
        
    except Exception as e:
        logger.error(
            "Affiliate program performance analysis failed",
            program_id=program_id,
            user_id=user_id,
            error=str(e)
        )
        raise
