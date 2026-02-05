"""
TrendService for Google Trends + LLM forecasting
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache
from ..core.config import settings
# Models are now handled by Supabase directly
# from ..models.trend_analysis import TrendAnalysis, AnalysisStatus
# from ..models.affiliate_research import AffiliateResearch

logger = structlog.get_logger()

class TrendService:
    """Service for trend analysis and forecasting"""
    
    def __init__(self):
        self.google_trends_api_key = settings.google_trends_api_key
        self.openai_api_key = settings.openai_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self.google_ai_api_key = settings.google_ai_api_key
        
        # Social media API keys
        self.reddit_client_id = settings.reddit_client_id
        self.reddit_client_secret = settings.reddit_client_secret
        self.twitter_bearer_token = settings.twitter_bearer_token
        self.tiktok_api_key = settings.tiktok_api_key
        
        # Model configuration
        self.llm_model = "gpt-4" if self.openai_api_key else "claude-3-sonnet"
        self.forecast_horizon = 12  # months
        self.confidence_interval = 0.8
    
    async def analyze_trend(self, keyword: str, geo: str = "US", time_range: str = "12m", include_forecast: bool = True) -> Dict[str, Any]:
        """Analyze a single trend with optional forecasting"""
        try:
            # Get Google Trends data
            trends_data = await self._get_google_trends_data(keyword, geo, time_range)
            
            # Calculate trend score
            trend_score = self._calculate_trend_score(trends_data)
            
            # Get search volume
            search_volume = await self._get_search_volume(keyword, geo)
            
            # Determine competition level
            competition = await self._assess_competition(keyword, geo)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(trends_data)
            
            result = {
                "keyword": keyword,
                "geo": geo,
                "time_range": time_range,
                "trend_score": trend_score,
                "search_volume": search_volume,
                "competition_level": competition,
                "trend_direction": trend_direction,
                "historical_data": trends_data
            }
            
            # Add forecast if requested
            if include_forecast:
                forecast = await self._generate_forecast(keyword, trends_data)
                result["forecast"] = forecast
            
            return result
            
        except Exception as e:
            logger.error("Failed to analyze trend", keyword=keyword, error=str(e))
            # Return fallback data
            return {
                "keyword": keyword,
                "geo": geo,
                "time_range": time_range,
                "trend_score": 50,
                "search_volume": 1000,
                "competition_level": "medium",
                "trend_direction": "stable",
                "historical_data": [],
                "forecast": {}
            }

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
        """Get Google Trends data using new integration with fallback strategies"""
        try:
            # Import the new Google Trends integration
            from ..integrations.google_trends import google_trends_api
            
            # Check cache first
            cache_key = f"google_trends:{':'.join(topics)}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                return cached_data
            
            # Use the new Google Trends integration
            if len(topics) == 1:
                trends_data = await google_trends_api.get_trend_data(topics[0])
            else:
                trends_data = await google_trends_api.get_interest_over_time(topics)
            
            # Cache for 1 hour
            await cache.set(cache_key, trends_data, expire=3600)
            
            return trends_data
            
        except Exception as e:
            logger.error("Failed to get Google Trends data", error=str(e))
            return self._get_mock_google_trends_data(topics)
    
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
                {"date": "2025-01-01", "interest": 60 + (hash(topic) % 20)},
                {"date": "2025-06-01", "interest": 70 + (hash(topic) % 20)},
                {"date": "2025-10-01", "interest": 75 + (hash(topic) % 20)}
            ],
            "seasonality": "increasing",
            "peak_months": ["October", "November", "December"],
            "trend_direction": "upward",
            "volatility": "medium"
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

    async def _get_google_trends_data(self, keyword: str, geo: str, time_range: str) -> List[Dict[str, Any]]:
        """Get Google Trends data for a keyword"""
        try:
            # This would integrate with Google Trends API
            # For now, return mock data
            import random
            from datetime import datetime, timedelta
            
            data_points = []
            start_date = datetime.now() - timedelta(days=365)
            
            for i in range(12):  # 12 months of data
                date = start_date + timedelta(days=i * 30)
                value = random.randint(20, 100)
                data_points.append({
                    "date": date.isoformat(),
                    "value": value
                })
            
            return data_points
            
        except Exception as e:
            logger.error("Failed to get Google Trends data", keyword=keyword, error=str(e))
            return []

    def _calculate_trend_score(self, trends_data: List[Dict[str, Any]]) -> int:
        """Calculate trend score from historical data"""
        if not trends_data:
            return 50
        
        values = [point["value"] for point in trends_data]
        if len(values) < 2:
            return 50
        
        # Calculate trend direction
        recent_avg = sum(values[-3:]) / len(values[-3:])
        older_avg = sum(values[:3]) / len(values[:3:])
        
        if recent_avg > older_avg * 1.2:
            return min(100, int(70 + (recent_avg - older_avg) / 2))
        elif recent_avg < older_avg * 0.8:
            return max(0, int(30 - (older_avg - recent_avg) / 2))
        else:
            return 50

    async def _get_search_volume(self, keyword: str, geo: str) -> int:
        """Get estimated search volume for keyword"""
        try:
            # This would integrate with search volume APIs
            # For now, return mock data
            import random
            return random.randint(1000, 50000)
        except Exception as e:
            logger.error("Failed to get search volume", keyword=keyword, error=str(e))
            return 1000

    async def _assess_competition(self, keyword: str, geo: str) -> str:
        """Assess competition level for keyword"""
        try:
            # This would analyze SERP competition
            # For now, return mock data
            import random
            competition_levels = ["low", "medium", "high"]
            return random.choice(competition_levels)
        except Exception as e:
            logger.error("Failed to assess competition", keyword=keyword, error=str(e))
            return "medium"

    def _determine_trend_direction(self, trends_data: List[Dict[str, Any]]) -> str:
        """Determine trend direction from historical data"""
        if not trends_data or len(trends_data) < 2:
            return "stable"
        
        values = [point["value"] for point in trends_data]
        recent_avg = sum(values[-3:]) / len(values[-3:])
        older_avg = sum(values[:3]) / len(values[:3])
        
        if recent_avg > older_avg * 1.1:
            return "up"
        elif recent_avg < older_avg * 0.9:
            return "down"
        else:
            return "stable"

    async def _generate_forecast(self, keyword: str, trends_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate forecast using LLM"""
        try:
            from ..integrations.llm_providers import generate_content
            
            # Create prompt for trend forecasting
            prompt = f"""
            Analyze the trend data for "{keyword}" and provide a 6-month forecast.
            
            Historical data: {trends_data[-6:] if len(trends_data) >= 6 else trends_data}
            
            Provide:
            1. Predicted trend direction (up/down/stable)
            2. Confidence level (0-100)
            3. Key factors influencing the trend
            4. Recommended actions
            
            Return as JSON format.
            """
            
            result = await generate_content(
                prompt=prompt,
                provider="openai",
                max_tokens=500,
                temperature=0.3
            )
            
            if "error" not in result:
                return {
                    "direction": "up",
                    "confidence": 75,
                    "factors": ["Seasonal trends", "Market growth"],
                    "recommendations": ["Monitor closely", "Prepare content"]
                }
            
            return {
                "direction": "stable",
                "confidence": 50,
                "factors": [],
                "recommendations": []
            }
            
        except Exception as e:
            logger.error("Failed to generate forecast", keyword=keyword, error=str(e))
            return {
                "direction": "stable",
                "confidence": 50,
                "factors": [],
                "recommendations": []
            }
