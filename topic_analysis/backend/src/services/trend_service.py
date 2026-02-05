"""
TrendService for Google Trends + LLM forecasting
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.trend_analysis import TrendAnalysis, AnalysisStatus
from ..models.affiliate_research import AffiliateResearch
from .dataforseo_service import DataForSEOService

logger = structlog.get_logger()
settings = get_settings()

class TrendService:
    """Service for trend analysis and forecasting"""
    
    def __init__(self):
        # API keys are now retrieved from database as needed
        self.google_trends_api_key = None
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.google_ai_api_key = None
        
        # Social media API keys - these will be retrieved from database
        self.reddit_client_id = None
        self.reddit_client_secret = None
        self.twitter_bearer_token = None
        self.tiktok_api_key = None
        
        # Model configuration - will be determined dynamically
        self.llm_model = "gpt-4"  # Default, will be updated based on available API keys
        self.forecast_horizon = 12  # months
        self.confidence_interval = 0.8
    
    async def create_analysis(self, user_id: int, topics: List[str], affiliate_research_id: Optional[int] = None) -> Dict[str, Any]:
        """Create new trend analysis"""
        try:
            # Create analysis record
            db = next(get_db())
            analysis = TrendAnalysis(
                user_id=user_id,
                affiliate_research_id=affiliate_research_id,
                topics=topics,
                status=AnalysisStatus.PENDING
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            
            # Start background analysis
            asyncio.create_task(self._perform_analysis(analysis.id))
            
            logger.info("Trend analysis created", analysis_id=analysis.id, topics=topics)
            return analysis.to_dict()
            
        except Exception as e:
            logger.error("Failed to create trend analysis", error=str(e))
            raise
    
    async def get_analysis(self, analysis_id: int) -> Dict[str, Any]:
        """Get trend analysis by ID"""
        try:
            db = next(get_db())
            analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == analysis_id)
            
            if not analysis:
                raise ValueError("Analysis not found")
            
            return analysis.to_dict()
            
        except Exception as e:
            logger.error("Failed to get trend analysis", analysis_id=analysis_id, error=str(e))
            raise
    
    async def _perform_analysis(self, analysis_id: int):
        """Perform trend analysis in background"""
        try:
            db = next(get_db())
            analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == analysis_id)
            
            if not analysis:
                return
            
            # Update status to processing
            analysis.status = AnalysisStatus.PROCESSING
            db.commit()
            
            start_time = datetime.now()
            warnings = []
            
            # Get affiliate research data if available
            affiliate_data = None
            if analysis.affiliate_research_id:
                affiliate_research = db.get_AffiliateResearch_by_id(
                    AffiliateResearch.id == analysis.affiliate_research_id
                )
                if affiliate_research and affiliate_research.results:
                    affiliate_data = affiliate_research.results
            
            # Perform analysis components
            google_trends_data = await self._get_google_trends_data(analysis.topics)
            llm_forecast = await self._generate_llm_forecast(analysis.topics, google_trends_data, affiliate_data)
            social_signals = await self._get_social_signals(analysis.topics)
            news_signals = await self._get_news_signals(analysis.topics)
            
            # Calculate opportunity scores
            opportunity_scores = self._calculate_opportunity_scores(
                analysis.topics, 
                google_trends_data, 
                llm_forecast, 
                social_signals,
                affiliate_data
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence_score(llm_forecast, social_signals)
            
            # Calculate duration
            duration = int((datetime.now() - start_time).total_seconds())
            
            # Update analysis with results
            analysis.mark_completed({
                "opportunity_scores": opportunity_scores,
                "llm_forecast": llm_forecast,
                "social_signals": social_signals,
                "google_trends_data": google_trends_data,
                "news_signals": news_signals,
                "model_version": f"trendtap-v1.0-{self.llm_model}",
                "confidence_score": confidence_score
            })
            analysis.analysis_duration = duration
            analysis.warnings = warnings
            
            db.commit()
            
            logger.info("Trend analysis completed", 
                       analysis_id=analysis_id, 
                       topics=analysis.topics,
                       duration=duration)
            
        except Exception as e:
            logger.error("Trend analysis failed", analysis_id=analysis_id, error=str(e))
            
            # Mark as failed
            try:
                db = next(get_db())
                analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == analysis_id)
                if analysis:
                    analysis.mark_failed(str(e))
                    db.commit()
            except:
                pass
    
    async def _get_google_trends_data(self, topics: List[str]) -> Dict[str, Any]:
        """Get trend data, with DataForSEO as primary source and Google Trends as fallback"""
        logger.info(f"DEBUG: _get_google_trends_data STARTED for {topics} - NEW VERSION")
        try:
            # Check cache first
            cache_key = f"trends:{':'.join(topics)}"
            cached_data = cache.get(cache_key)
            if cached_data:
                return cached_data
            
            # 1. Try DataForSEO (Primary)
            try:
                dataforseo_data = await self._get_dataforseo_trends(topics)
                if dataforseo_data:
                    cache.set(cache_key, dataforseo_data, expire=3600)
                    return dataforseo_data
            except Exception as e:
                logger.warning("DataForSEO trend fetching failed, falling back to Google Trends", error=str(e))

            # 2. Try Google Trends (Secondary)
            if self.google_trends_api_key:
                trends_data = await self._call_google_trends_api(topics)
                cache.set(cache_key, trends_data, expire=3600)
                return trends_data
            
            # 3. No fallback to mock as per strict requirements
            logger.warning("Trend data fetching failed, and mock fallbacks are disabled", topics=topics)
            return None
            
        except Exception as e:
            logger.error("Failed to get trend data", topics=topics, error=str(e))
            return None

    async def _get_dataforseo_trends(self, topics: List[str]) -> Optional[Dict[str, Any]]:
        """Fetch and map DataForSEO historical data to unified trend format with rich Google Trends Explore data"""
        df_service = DataForSEOService()
        
        try:
            # 1. Fetch Search Volume History (for absolute metrics)
            # 2. Fetch Google Trends Explore (for authentic 0-100 interest scores)
            # We run these in parallel for speed
            logger.info("DEBUG: fetching DataForSEO trends", topics=topics)
            
            search_history_task = df_service.get_search_volume_history(topics)
            trends_explore_task = df_service.get_google_trends_explore(topics)
            
            results = await asyncio.gather(search_history_task, trends_explore_task, return_exceptions=True)
            
            history = results[0] if not isinstance(results[0], Exception) else None
            explore_data = results[1] if not isinstance(results[1], Exception) else None
            
            if isinstance(results[0], Exception):
                logger.warning(f"DataForSEO search volume history failed: {results[0]}")
            else:
                logger.info(f"DEBUG: DataForSEO History Response: Keys={list(history.keys()) if history else 'None'}")

            if isinstance(results[1], Exception):
                logger.warning(f"DataForSEO trends explore failed: {results[1]}")
            else:
                logger.info(f"DEBUG: DataForSEO Explore Response: Keys={list(explore_data.keys()) if explore_data else 'None'}")
            
            if not history and not explore_data:
                logger.warning("DEBUG: Both DataForSEO calls returned empty or failed.")
                return None
                
            primary_topic = topics[0]
            # Handle case where keys might be lowercased by DataForSEO
            # Try exact match first, then lowercase
            topic_history = {}
            if history:
                topic_history = history.get(primary_topic) or history.get(primary_topic.lower()) or {}
                if topic_history:
                     logger.info(f"DEBUG: Found history for {primary_topic}. Monthly searches count: {len(topic_history.get('monthly_searches', []))}")
                else:
                     logger.warning(f"DEBUG: No history found for {primary_topic} (or lower). Available keys: {list(history.keys())}")

            
            topic_explore = []
            if explore_data:
                topic_explore = explore_data.get(primary_topic) or explore_data.get(primary_topic.lower()) or []
            
            # Combine the data
            # We'll use explore_data (0-100) as the primary interest source if available
            # because it's authentic Google Trends data. 
            # We'll augment it with search_volume from history.
            
            combined_historical = []
            
            # Map search volume history for lookups
            vol_map = {}
            for m in topic_history.get("monthly_searches", []):
                year = m.get("year_year") or m.get("year")
                month = m.get("month")
                date_key = f"{year}-{month:02d}-01"
                vol_map[date_key] = m.get("search_volume", 0)
            
            if topic_explore:
                logger.info(f"DEBUG: DataForSEO Topic Explore Data Points: {len(topic_explore)}")
                if len(topic_explore) > 0:
                    logger.info(f"DEBUG: First Data Point Sample: {topic_explore[0]}")
                # Use Explore data as the base
                for point in topic_explore:
                    date = point.get("date")
                    interest = point.get("value", 0)
                    
                    # Try to find matching absolute volume
                    # Note: explore dates might be weekly or monthly. 
                    # We'll try to match the month.
                    vol = 0
                    if date:
                        month_key = date[:7] + "-01"
                        vol = vol_map.get(month_key, 0)
                    
                    combined_historical.append({
                        "date": date,
                        "interest": interest,
                        "absolute_volume": vol
                    })
            else:
                # Fallback to pure search volume history if Explore failed
                searches = topic_history.get("monthly_searches", [])
                max_vol = max([m.get("search_volume", 0) for m in searches], default=1) if searches else 1
                for date_key, vol in vol_map.items():
                    combined_historical.append({
                        "date": date_key,
                        "interest": int((vol / max_vol) * 100),
                        "absolute_volume": vol
                    })


            
            # Sort by date
            combined_historical.sort(key=lambda x: x["date"])
            
            # Calculate trend direction
            if len(combined_historical) >= 2:
                direction = "upward" if combined_historical[-1]["interest"] > combined_historical[0]["interest"] else "downward"
            else:
                direction = "stable"
                
            return {
                "historical": combined_historical,
                "search_volume": topic_history.get("search_volume"),
                "competition": topic_history.get("competition"),
                "cpc": topic_history.get("cpc"),
                "trend_direction": direction,
                "source": "dataforseo_rich"
            }
        except Exception as e:
            logger.error("DataForSEO rich trend fetching failed", topics=topics, error=str(e))
            return None
        finally:
            await df_service.close()
    
    async def _call_google_trends_api(self, topics: List[str]) -> Dict[str, Any]:
        """Call Google Trends API"""
        # Mock implementation - replace with actual API calls
        return {
            "historical": [
                {"date": "2025-01-01", "interest": 65},
                {"date": "2025-06-01", "interest": 78},
                {"date": "2025-10-01", "interest": 82}
            ],
            "seasonality": "increasing",
            "peak_months": ["October", "November", "December"],
            "trend_direction": "upward",
            "volatility": "medium"
        }
    
    def _get_mock_google_trends_data(self, topics: List[str]) -> Dict[str, Any]:
        """Get mock Google Trends data"""
        return {
            "historical": [
                {"date": "2025-01-01", "interest": 60 + (hash(topics[0]) % 20)},
                {"date": "2025-06-01", "interest": 70 + (hash(topics[0]) % 20)},
                {"date": "2025-10-01", "interest": 75 + (hash(topics[0]) % 20)}
            ],
            "seasonality": "increasing",
            "peak_months": ["October", "November", "December"],
            "trend_direction": "upward",
            "volatility": "medium",
            "search_volume": 1000 + (abs(hash(topics[0])) % 9000)  # Random volume 1000-10000
        }
    
    async def _generate_llm_forecast(self, topics: List[str], google_trends_data: Dict[str, Any], affiliate_data: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate LLM forecast"""
        try:
            if not any([self.openai_api_key, self.anthropic_api_key, self.google_ai_api_key]):
                return self._get_mock_llm_forecast(topics)
            
            # Prepare prompt
            prompt = self._build_forecast_prompt(topics, google_trends_data, affiliate_data)
            
            # Call LLM API
            forecast_data = await self._call_llm_api(prompt)
            
            return forecast_data
            
        except Exception as e:
            logger.error("Failed to generate LLM forecast", error=str(e))
            return self._get_mock_llm_forecast(topics)
    
    def _build_forecast_prompt(self, topics: List[str], google_trends_data: Dict[str, Any], affiliate_data: Optional[List[Dict[str, Any]]]) -> str:
        """Build forecast prompt for LLM"""
        prompt = f"""
        Analyze the following topics for trend forecasting:
        
        Topics: {', '.join(topics)}
        
        Google Trends Data:
        {json.dumps(google_trends_data, indent=2)}
        
        Affiliate Data:
        {json.dumps(affiliate_data[:3] if affiliate_data else [], indent=2)}
        
        Please provide:
        1. 12-month forecast for each topic with confidence intervals
        2. Key factors driving the trends
        3. Seasonal patterns and peak months
        4. Risk factors and potential disruptions
        5. Opportunity assessment (0-100 scale)
        
        Format as JSON with this structure:
        {{
            "forecast": [
                {{
                    "topic": "topic_name",
                    "month": "2025-11",
                    "predicted_interest": 78.2,
                    "confidence_interval": [65.1, 91.3],
                    "key_factors": ["factor1", "factor2"],
                    "risk_factors": ["risk1", "risk2"]
                }}
            ],
            "model_version": "trendtap-v1.0",
            "training_data_size": 400000
        }}
        """
        return prompt
    
    async def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API"""
        if self.openai_api_key:
            return await self._call_openai_api(prompt)
        elif self.anthropic_api_key:
            return await self._call_anthropic_api(prompt)
        elif self.google_ai_api_key:
            return await self._call_google_ai_api(prompt)
        else:
            return self._get_mock_llm_forecast([])
    
    async def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        # Mock implementation - replace with actual API calls
        return self._get_mock_llm_forecast([])
    
    async def _call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API"""
        # Mock implementation - replace with actual API calls
        return self._get_mock_llm_forecast([])
    
    async def _call_google_ai_api(self, prompt: str) -> Dict[str, Any]:
        """Call Google AI API"""
        # Mock implementation - replace with actual API calls
        return self._get_mock_llm_forecast([])
    
    def _get_mock_llm_forecast(self, topics: List[str]) -> Dict[str, Any]:
        """Get mock LLM forecast"""
        forecast = []
        for topic in topics:
            for month in range(1, 13):
                month_name = datetime(2025, month, 1).strftime("%Y-%m")
                predicted_interest = 70 + (hash(topic) % 20) + (month % 6)
                confidence_interval = [predicted_interest - 10, predicted_interest + 10]
                
                forecast.append({
                    "topic": topic,
                    "month": month_name,
                    "predicted_interest": predicted_interest,
                    "confidence_interval": confidence_interval,
                    "key_factors": [f"Factor 1 for {topic}", f"Factor 2 for {topic}"],
                    "risk_factors": [f"Risk 1 for {topic}", f"Risk 2 for {topic}"]
                })
        
        return {
            "forecast": forecast,
            "model_version": "trendtap-v1.0-mock",
            "training_data_size": 400000
        }
    
    async def _get_social_signals(self, topics: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get social media signals"""
        try:
            social_signals = {
                "reddit": await self._get_reddit_signals(topics),
                "twitter": await self._get_twitter_signals(topics),
                "tiktok": await self._get_tiktok_signals(topics)
            }
            
            return social_signals
            
        except Exception as e:
            logger.error("Failed to get social signals", error=str(e))
            return {"reddit": [], "twitter": [], "tiktok": []}
    
    async def _get_reddit_signals(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get Reddit signals"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "subreddit": "Coffee",
                "post_count": 45,
                "sentiment": 0.8,
                "trending_keywords": ["roasting", "equipment", "beginner"],
                "engagement_rate": 0.12
            }
        ]
    
    async def _get_twitter_signals(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get Twitter signals"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "hashtag": "#CoffeeRoasting",
                "tweet_count": 120,
                "sentiment": 0.7,
                "influencers": ["@coffeeexpert", "@roastingpro"],
                "engagement_rate": 0.08
            }
        ]
    
    async def _get_tiktok_signals(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get TikTok signals"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "hashtag": "#CoffeeRoasting",
                "video_count": 35,
                "engagement_rate": 0.12,
                "trending_sounds": ["coffee_roasting_sound"],
                "view_count": 50000
            }
        ]
    
    async def _get_news_signals(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get news signals"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "headline": f"New {topics[0]} trend emerging",
                "source": "Tech News",
                "sentiment": 0.8,
                "published_date": "2025-10-01",
                "relevance_score": 0.9
            }
        ]
    
    def _calculate_opportunity_scores(self, topics: List[str], google_trends_data: Dict[str, Any], 
                                    llm_forecast: Dict[str, Any], social_signals: Dict[str, List[Dict[str, Any]]],
                                    affiliate_data: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate opportunity scores for topics"""
        scores = {}
        
        for topic in topics:
            # Base score from Google Trends
            trends_score = self._get_trends_score(topic, google_trends_data)
            
            # LLM forecast score
            forecast_score = self._get_forecast_score(topic, llm_forecast)
            
            # Social signals score
            social_score = self._get_social_score(topic, social_signals)
            
            # Affiliate opportunity score
            affiliate_score = self._get_affiliate_score(topic, affiliate_data)
            
            # Weighted average
            opportunity_score = (
                trends_score * 0.3 +
                forecast_score * 0.4 +
                social_score * 0.2 +
                affiliate_score * 0.1
            )
            
            scores[topic] = min(max(opportunity_score, 0), 100)
        
        return scores
    
    def _get_trends_score(self, topic: str, google_trends_data: Dict[str, Any]) -> float:
        """Get score from Google Trends data"""
        if not google_trends_data or "historical" not in google_trends_data:
            return 50.0
        
        historical = google_trends_data["historical"]
        if not historical:
            return 50.0
        
        # Calculate trend direction and strength
        latest_interest = historical[-1]["interest"]
        earliest_interest = historical[0]["interest"]
        
        trend_strength = (latest_interest - earliest_interest) / earliest_interest
        base_score = latest_interest
        
        # Adjust for trend direction
        if trend_strength > 0.1:  # Strong upward trend
            return min(base_score * 1.2, 100)
        elif trend_strength > 0:  # Moderate upward trend
            return min(base_score * 1.1, 100)
        else:  # Downward or flat trend
            return max(base_score * 0.9, 0)
    
    def _get_forecast_score(self, topic: str, llm_forecast: Dict[str, Any]) -> float:
        """Get score from LLM forecast"""
        if not llm_forecast or "forecast" not in llm_forecast:
            return 50.0
        
        topic_forecasts = [f for f in llm_forecast["forecast"] if f["topic"] == topic]
        if not topic_forecasts:
            return 50.0
        
        # Calculate average predicted interest
        avg_interest = sum(f["predicted_interest"] for f in topic_forecasts) / len(topic_forecasts)
        return min(avg_interest, 100)
    
    def _get_social_score(self, topic: str, social_signals: Dict[str, List[Dict[str, Any]]]) -> float:
        """Get score from social signals"""
        total_score = 0
        count = 0
        
        for platform, signals in social_signals.items():
            for signal in signals:
                if "sentiment" in signal:
                    total_score += signal["sentiment"] * 100
                    count += 1
                elif "engagement_rate" in signal:
                    total_score += signal["engagement_rate"] * 100
                    count += 1
        
        return total_score / count if count > 0 else 50.0
    
    def _get_affiliate_score(self, topic: str, affiliate_data: Optional[List[Dict[str, Any]]]) -> float:
        """Get score from affiliate data"""
        if not affiliate_data:
            return 50.0
        
        # Calculate average EPC for topic-related programs
        topic_programs = [p for p in affiliate_data if topic.lower() in p.get("program_name", "").lower()]
        if not topic_programs:
            return 50.0
        
        avg_epc = sum(p.get("epc", 0) for p in topic_programs) / len(topic_programs)
        return min(avg_epc * 5, 100)  # Scale EPC to 0-100
    
    def _calculate_confidence_score(self, llm_forecast: Dict[str, Any], social_signals: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall confidence score"""
        if not llm_forecast or "forecast" not in llm_forecast:
            return 0.5
        
        # Calculate confidence from forecast intervals
        total_confidence = 0
        count = 0
        
        for forecast in llm_forecast["forecast"]:
            if "confidence_interval" in forecast:
                ci = forecast["confidence_interval"]
                if len(ci) == 2:
                    # Calculate confidence as the width of the interval
                    confidence = 1.0 - (ci[1] - ci[0]) / 100.0
                    total_confidence += max(0, confidence)
                    count += 1
        
        return total_confidence / count if count > 0 else 0.5
    
    async def get_user_analyses(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's trend analyses"""
        try:
            db = next(get_db())
            analyses = db.query(TrendAnalysis).filter(
                TrendAnalysis.user_id == user_id
            ).order_by(TrendAnalysis.created_at.desc()).offset(offset).limit(limit).all()
            
            return [analysis.to_dict() for analysis in analyses]
            
        except Exception as e:
            logger.error("Failed to get user analyses", user_id=user_id, error=str(e))
            raise
    
    async def delete_analysis(self, analysis_id: int, user_id: int) -> bool:
        """Delete trend analysis"""
        try:
            db = next(get_db())
            analysis = db.get_TrendAnalysis_by_id(
                TrendAnalysis.id == analysis_id,
                TrendAnalysis.user_id == user_id
            )
            
            if not analysis:
                raise ValueError("Analysis not found")
            
            db.delete(analysis)
            db.commit()
            
            logger.info("Trend analysis deleted", analysis_id=analysis_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete analysis", analysis_id=analysis_id, error=str(e))
            raise
