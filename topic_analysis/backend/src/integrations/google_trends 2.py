"""
Google Trends Integration
Provides trend data analysis with multiple fallback strategies:
1. Future Google Trends API (when available)
2. LLM-based trend analysis
3. CSV upload from manual Google Trends searches
"""

import asyncio
import json
import logging
import csv
import io
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
from ..core.config import settings

logger = logging.getLogger(__name__)

class GoogleTrendsAPI:
    """Google Trends integration with multiple fallback strategies"""
    
    def __init__(self):
        self.timeout = 30.0
        self.use_llm_fallback = getattr(settings, 'USE_LLM_TRENDS_FALLBACK', True)
        self.llm_provider = getattr(settings, 'LLM_TRENDS_PROVIDER', 'openai')
        self.openai_api_key = getattr(settings, 'OPENAI_API_KEY', None)
        self.anthropic_api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
        self.google_ai_api_key = getattr(settings, 'GOOGLE_AI_API_KEY', None)
        
    async def get_trend_data(
        self,
        keyword: str,
        geo: str = "US",
        timeframe: str = "today 12-m",
        category: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch trend data for a specific keyword using fallback strategies
        
        Args:
            keyword: The keyword to search for
            geo: Geographic location (e.g., "US", "GB", "CA")
            timeframe: Time period for data (e.g., "today 12-m", "today 5-y")
            category: Category ID (0 = all categories)
            
        Returns:
            Dict containing trend data
        """
        try:
            # Strategy 1: Try future Google Trends API (placeholder)
            if await self._is_google_trends_api_available():
                return await self._call_future_google_trends_api(keyword, geo, timeframe, category)
            
            # Strategy 2: Use LLM-based trend analysis
            elif self.use_llm_fallback and self._has_llm_credentials():
                logger.info(f"Using LLM fallback for trend analysis: {keyword}")
                return await self._analyze_trends_with_llm(keyword, geo, timeframe)
            
            # Strategy 3: Return mock data with instructions for CSV upload
            else:
                logger.info(f"Using mock data with CSV upload instructions for: {keyword}")
                return self._get_mock_trend_data_with_instructions(keyword, geo, timeframe)
                
        except Exception as e:
            logger.error(f"Error fetching trend data for {keyword}: {e}")
            return self._get_mock_trend_data_with_instructions(keyword, geo, timeframe)
    
    async def get_related_queries(
        self,
        keyword: str,
        geo: str = "US"
    ) -> List[Dict[str, Any]]:
        """
        Get related queries for a keyword using fallback strategies
        
        Args:
            keyword: The keyword to get related queries for
            geo: Geographic location
            
        Returns:
            List of related queries with their search volumes
        """
        try:
            # Strategy 1: Try future Google Trends API
            if await self._is_google_trends_api_available():
                return await self._call_future_google_trends_related_queries(keyword, geo)
            
            # Strategy 2: Use LLM-based related queries
            elif self.use_llm_fallback and self._has_llm_credentials():
                return await self._generate_related_queries_with_llm(keyword, geo)
            
            # Strategy 3: Return mock data
            else:
                return self._get_mock_related_queries(keyword)
                
        except Exception as e:
            logger.error(f"Error fetching related queries for {keyword}: {e}")
            return self._get_mock_related_queries(keyword)
    
    async def get_interest_over_time(
        self,
        keywords: List[str],
        geo: str = "US",
        timeframe: str = "today 12-m"
    ) -> Dict[str, Any]:
        """
        Get interest over time for multiple keywords using fallback strategies
        
        Args:
            keywords: List of keywords to compare
            geo: Geographic location
            timeframe: Time period for data
            
        Returns:
            Dict containing interest over time data
        """
        try:
            # Strategy 1: Try future Google Trends API
            if await self._is_google_trends_api_available():
                return await self._call_future_google_trends_interest_over_time(keywords, geo, timeframe)
            
            # Strategy 2: Use LLM-based analysis
            elif self.use_llm_fallback and self._has_llm_credentials():
                return await self._analyze_interest_over_time_with_llm(keywords, geo, timeframe)
            
            # Strategy 3: Return mock data
            else:
                return self._get_mock_interest_over_time(keywords)
                
        except Exception as e:
            logger.error(f"Error fetching interest over time: {e}")
            return self._get_mock_interest_over_time(keywords)
    
    async def process_csv_upload(self, csv_content: str, keyword: str) -> Dict[str, Any]:
        """
        Process CSV data uploaded from manual Google Trends searches
        
        Args:
            csv_content: CSV content from Google Trends export
            keyword: The keyword this data is for
            
        Returns:
            Processed trend data from CSV
        """
        try:
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(csv_reader)
            
            if not rows:
                raise ValueError("CSV file is empty or invalid")
            
            # Process the CSV data
            processed_data = self._process_csv_trend_data(rows, keyword)
            
            logger.info(f"Successfully processed CSV data for keyword: {keyword}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing CSV data for {keyword}: {e}")
            raise ValueError(f"Failed to process CSV data: {e}")
    
    async def _is_google_trends_api_available(self) -> bool:
        """Check if future Google Trends API is available (placeholder)"""
        # This will be updated when Google releases the new API
        return False
    
    async def _call_future_google_trends_api(self, keyword: str, geo: str, timeframe: str, category: int) -> Dict[str, Any]:
        """Placeholder for future Google Trends API calls"""
        # This will be implemented when Google releases the new API
        raise NotImplementedError("Google Trends API not yet available")
    
    async def _call_future_google_trends_related_queries(self, keyword: str, geo: str) -> List[Dict[str, Any]]:
        """Placeholder for future Google Trends related queries API"""
        raise NotImplementedError("Google Trends API not yet available")
    
    async def _call_future_google_trends_interest_over_time(self, keywords: List[str], geo: str, timeframe: str) -> Dict[str, Any]:
        """Placeholder for future Google Trends interest over time API"""
        raise NotImplementedError("Google Trends API not yet available")
    
    def _has_llm_credentials(self) -> bool:
        """Check if LLM credentials are available"""
        return any([
            self.openai_api_key,
            self.anthropic_api_key,
            self.google_ai_api_key
        ])
    
    async def _analyze_trends_with_llm(self, keyword: str, geo: str, timeframe: str) -> Dict[str, Any]:
        """Analyze trends using LLM when Google Trends API is not available"""
        try:
            # Create a prompt for trend analysis
            prompt = self._build_trend_analysis_prompt(keyword, geo, timeframe)
            
            # Call LLM
            llm_response = await self._call_llm(prompt)
            
            # Parse LLM response
            trend_data = self._parse_llm_trend_response(llm_response, keyword)
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error in LLM trend analysis for {keyword}: {e}")
            return self._get_mock_trend_data_with_instructions(keyword, geo, timeframe)
    
    async def _generate_related_queries_with_llm(self, keyword: str, geo: str) -> List[Dict[str, Any]]:
        """Generate related queries using LLM"""
        try:
            prompt = f"""
            Generate 10 related search queries for the keyword "{keyword}" in {geo}.
            Include both top queries (high volume, stable) and rising queries (growing interest).
            Return as JSON with this format:
            {{
                "top_queries": [{{"query": "query text", "value": 85, "formatted_value": "85"}}],
                "rising_queries": [{{"query": "query text", "value": 70, "formatted_value": "70"}}]
            }}
            """
            
            llm_response = await self._call_llm(prompt)
            related_queries = self._parse_llm_related_queries_response(llm_response)
            
            return related_queries
            
        except Exception as e:
            logger.error(f"Error generating related queries with LLM for {keyword}: {e}")
            return self._get_mock_related_queries(keyword)
    
    async def _analyze_interest_over_time_with_llm(self, keywords: List[str], geo: str, timeframe: str) -> Dict[str, Any]:
        """Analyze interest over time using LLM"""
        try:
            prompt = f"""
            Analyze search interest trends for these keywords: {', '.join(keywords)} in {geo} over {timeframe}.
            Generate realistic monthly data points showing relative search volumes.
            Return as JSON with this format:
            {{
                "timeline": [
                    {{"date": "2024-01-01", "values": {{"keyword1": 50, "keyword2": 60}}}},
                    {{"date": "2024-02-01", "values": {{"keyword1": 55, "keyword2": 65}}}}
                ],
                "comparison": {{
                    "keyword1": {{"average": 60, "peak": 80, "total": 720}},
                    "keyword2": {{"average": 70, "peak": 90, "total": 840}}
                }}
            }}
            """
            
            llm_response = await self._call_llm(prompt)
            interest_data = self._parse_llm_interest_over_time_response(llm_response, keywords)
            
            return interest_data
            
        except Exception as e:
            logger.error(f"Error analyzing interest over time with LLM: {e}")
            return self._get_mock_interest_over_time(keywords)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider"""
        try:
            if self.llm_provider == 'openai' and self.openai_api_key:
                return await self._call_openai(prompt)
            elif self.llm_provider == 'anthropic' and self.anthropic_api_key:
                return await self._call_anthropic(prompt)
            elif self.llm_provider == 'google' and self.google_ai_api_key:
                return await self._call_google_ai(prompt)
            else:
                raise ValueError("No valid LLM provider configured")
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
    
    async def _call_google_ai(self, prompt: str) -> str:
        """Call Google AI API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.google_ai_api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 2000,
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    def _build_trend_analysis_prompt(self, keyword: str, geo: str, timeframe: str) -> str:
        """Build prompt for trend analysis"""
        return f"""
        Analyze search trends for the keyword "{keyword}" in {geo} over {timeframe}.
        
        Generate realistic trend data including:
        1. Monthly interest values (0-100 scale)
        2. Trend direction (rising/falling/stable)
        3. Related queries (top and rising)
        4. Summary statistics
        
        Return as JSON with this exact format:
        {{
            "keyword": "{keyword}",
            "timestamp": "2024-01-01T00:00:00Z",
            "interest_over_time": [
                {{"date": "2024-01-01", "value": 50}},
                {{"date": "2024-02-01", "value": 65}}
            ],
            "related_queries": [],
            "rising_queries": [
                {{"query": "{keyword} guide", "value": 80, "formatted_value": "80"}},
                {{"query": "best {keyword}", "value": 75, "formatted_value": "75"}}
            ],
            "top_queries": [
                {{"query": "how to {keyword}", "value": 90, "formatted_value": "90"}},
                {{"query": "{keyword} tips", "value": 85, "formatted_value": "85"}}
            ],
            "summary": {{
                "average_interest": 75.0,
                "peak_interest": 95,
                "trend_direction": "rising",
                "volatility": 15.2
            }}
        }}
        """
    
    def _parse_llm_trend_response(self, response: str, keyword: str) -> Dict[str, Any]:
        """Parse LLM response for trend data"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['keyword', 'interest_over_time', 'summary']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing LLM trend response: {e}")
            return self._get_mock_trend_data_with_instructions(keyword, "US", "today 12-m")
    
    def _parse_llm_related_queries_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for related queries"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            related_queries = []
            
            # Add top queries
            if 'top_queries' in data:
                for query in data['top_queries']:
                    related_queries.append({
                        "query": query.get('query', ''),
                        "value": query.get('value', 0),
                        "formatted_value": str(query.get('value', 0)),
                        "link": "top"
                    })
            
            # Add rising queries
            if 'rising_queries' in data:
                for query in data['rising_queries']:
                    related_queries.append({
                        "query": query.get('query', ''),
                        "value": query.get('value', 0),
                        "formatted_value": str(query.get('value', 0)),
                        "link": "rising"
                    })
            
            return related_queries
            
        except Exception as e:
            logger.error(f"Error parsing LLM related queries response: {e}")
            return []
    
    def _parse_llm_interest_over_time_response(self, response: str, keywords: List[str]) -> Dict[str, Any]:
        """Parse LLM response for interest over time"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing LLM interest over time response: {e}")
            return self._get_mock_interest_over_time(keywords)
    
    def _process_csv_trend_data(self, rows: List[Dict[str, str]], keyword: str) -> Dict[str, Any]:
        """Process CSV data from Google Trends export"""
        try:
            processed = {
                "keyword": keyword,
                "timestamp": datetime.utcnow().isoformat(),
                "interest_over_time": [],
                "related_queries": [],
                "rising_queries": [],
                "top_queries": [],
                "summary": {
                    "average_interest": 0,
                    "peak_interest": 0,
                    "trend_direction": "stable",
                    "volatility": 0
                },
                "data_source": "csv_upload"
            }
            
            interest_values = []
            
            # Process each row
            for row in rows:
                # Handle different CSV formats from Google Trends
                date_key = None
                value_key = None
                
                # Find date and value columns
                for col in row.keys():
                    col_lower = col.lower()
                    if 'date' in col_lower or 'time' in col_lower:
                        date_key = col
                    elif 'value' in col_lower or 'interest' in col_lower or keyword.lower() in col_lower:
                        value_key = col
                
                if date_key and value_key:
                    try:
                        date_str = row[date_key]
                        value = int(float(row[value_key]))
                        
                        processed["interest_over_time"].append({
                            "date": date_str,
                            "value": value
                        })
                        interest_values.append(value)
                    except (ValueError, TypeError):
                        continue
            
            # Calculate summary statistics
            if interest_values:
                processed["summary"]["average_interest"] = sum(interest_values) / len(interest_values)
                processed["summary"]["peak_interest"] = max(interest_values)
                
                # Determine trend direction
                if len(interest_values) >= 2:
                    recent_avg = sum(interest_values[-3:]) / min(3, len(interest_values))
                    older_avg = sum(interest_values[:3]) / min(3, len(interest_values))
                    
                    if recent_avg > older_avg * 1.1:
                        processed["summary"]["trend_direction"] = "rising"
                    elif recent_avg < older_avg * 0.9:
                        processed["summary"]["trend_direction"] = "falling"
                
                # Calculate volatility
                variance = sum((x - processed["summary"]["average_interest"]) ** 2 for x in interest_values) / len(interest_values)
                processed["summary"]["volatility"] = variance ** 0.5
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing CSV trend data: {e}")
            raise ValueError(f"Failed to process CSV data: {e}")
    
    def _get_mock_trend_data_with_instructions(self, keyword: str, geo: str, timeframe: str) -> Dict[str, Any]:
        """Get mock data with instructions for CSV upload"""
        return {
            "keyword": keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "interest_over_time": [
                {"date": "2024-01-01", "value": 50},
                {"date": "2024-02-01", "value": 65},
                {"date": "2024-03-01", "value": 70},
                {"date": "2024-04-01", "value": 80},
                {"date": "2024-05-01", "value": 75},
                {"date": "2024-06-01", "value": 85},
                {"date": "2024-07-01", "value": 90},
                {"date": "2024-08-01", "value": 88},
                {"date": "2024-09-01", "value": 92},
                {"date": "2024-10-01", "value": 95},
                {"date": "2024-11-01", "value": 98},
                {"date": "2024-12-01", "value": 100}
            ],
            "related_queries": [],
            "rising_queries": [
                {"query": f"{keyword} guide", "value": 80, "formatted_value": "80"},
                {"query": f"best {keyword}", "value": 75, "formatted_value": "75"},
                {"query": f"{keyword} 2024", "value": 70, "formatted_value": "70"}
            ],
            "top_queries": [
                {"query": f"how to {keyword}", "value": 90, "formatted_value": "90"},
                {"query": f"{keyword} tips", "value": 85, "formatted_value": "85"},
                {"query": f"{keyword} review", "value": 80, "formatted_value": "80"}
            ],
            "summary": {
                "average_interest": 80.0,
                "peak_interest": 100,
                "trend_direction": "rising",
                "volatility": 15.2
            },
            "data_source": "mock_with_instructions",
            "instructions": {
                "message": "Google Trends API not available. Upload CSV data for real analysis.",
                "steps": [
                    "1. Go to https://trends.google.com/trends/",
                    f"2. Search for '{keyword}'",
                    f"3. Set location to {geo}",
                    f"4. Set time range to {timeframe}",
                    "5. Click 'Download' and select CSV",
                    "6. Upload the CSV file here for real trend analysis"
                ],
                "csv_format": "Date,Interest Value",
                "example_csv": "2024-01-01,50\n2024-02-01,65\n2024-03-01,70"
            }
        }
    
    def _get_mock_related_queries(self, keyword: str) -> List[Dict[str, Any]]:
        """Get mock related queries"""
        return [
            {"query": f"{keyword} guide", "value": 80, "formatted_value": "80", "link": "top"},
            {"query": f"best {keyword}", "value": 75, "formatted_value": "75", "link": "top"},
            {"query": f"{keyword} 2024", "value": 70, "formatted_value": "70", "link": "rising"}
        ]
    
    def _get_mock_interest_over_time(self, keywords: List[str]) -> Dict[str, Any]:
        """Get mock interest over time data"""
        return {
            "keywords": keywords,
            "timestamp": datetime.utcnow().isoformat(),
            "timeline": [
                {
                    "date": "2024-01-01",
                    "values": {keyword: 50 + i * 10 for i, keyword in enumerate(keywords)}
                },
                {
                    "date": "2024-02-01",
                    "values": {keyword: 60 + i * 10 for i, keyword in enumerate(keywords)}
                },
                {
                    "date": "2024-03-01",
                    "values": {keyword: 70 + i * 10 for i, keyword in enumerate(keywords)}
                }
            ],
            "comparison": {
                keyword: {
                    "average": 60 + i * 10,
                    "peak": 80 + i * 10,
                    "total": 180 + i * 30
                } for i, keyword in enumerate(keywords)
            },
            "data_source": "mock"
        }

# Global instance
google_trends_api = GoogleTrendsAPI()

# Convenience functions
async def get_trend_data(keyword: str, geo: str = "US", timeframe: str = "today 12-m") -> Dict[str, Any]:
    """Get trend data for a keyword"""
    return await google_trends_api.get_trend_data(keyword, geo, timeframe)

async def get_related_queries(keyword: str, geo: str = "US") -> List[Dict[str, Any]]:
    """Get related queries for a keyword"""
    return await google_trends_api.get_related_queries(keyword, geo)

async def get_interest_over_time(keywords: List[str], geo: str = "US", timeframe: str = "today 12-m") -> Dict[str, Any]:
    """Get interest over time for multiple keywords"""
    return await google_trends_api.get_interest_over_time(keywords, geo, timeframe)

async def process_csv_upload(csv_content: str, keyword: str) -> Dict[str, Any]:
    """Process CSV data uploaded from manual Google Trends searches"""
    return await google_trends_api.process_csv_upload(csv_content, keyword)