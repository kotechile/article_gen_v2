"""
External Tool Integration Service
Handles integration with Semrush, Ahrefs, and Ubersuggest APIs
"""

import uuid
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ..models.external_tool_result import ExternalToolResult
from ..services.api_key_service import APIKeyService, ServiceName
from ..core.redis import cache

logger = structlog.get_logger()

class ExternalToolIntegrationService:
    """Service for integrating with external keyword research tools"""
    
    def __init__(self, db: Session):
        self.db = db
        self.api_key_service = APIKeyService(db)
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.rate_limit_delay = 1  # 1 second between requests
        
        # API endpoints
        self.semrush_url = "https://api.semrush.com"
        self.ahrefs_url = "https://apiv2.ahrefs.com"
        self.ubersuggest_url = "https://app.neilpatel.com/api"
        
        # Request headers
        self.headers = {
            'User-Agent': 'TrendTap/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    async def process_external_tool_request(
        self,
        user_id: str,
        workflow_session_id: str,
        tool_name: str,
        query_type: str,
        query_parameters: Dict[str, Any],
        seed_keywords: List[str],
        trend_analysis_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process external tool request and store results"""
        try:
            logger.info("Processing external tool request", 
                       user_id=user_id, 
                       tool_name=tool_name,
                       query_type=query_type)
            
            # Create external tool result record
            result_id = str(uuid.uuid4())
            external_result = ExternalToolResult(
                id=result_id,
                user_id=user_id,
                workflow_session_id=workflow_session_id,
                trend_analysis_id=trend_analysis_id,
                tool_name=tool_name,
                query_type=query_type,
                query_parameters=query_parameters,
                seed_keywords=seed_keywords,
                status="pending"
            )
            
            self.db.add(external_result)
            self.db.commit()
            self.db.refresh(external_result)
            
            # Process based on tool
            if tool_name == "semrush":
                result_data = await self._process_semrush_request(user_id, query_type, query_parameters, seed_keywords)
            elif tool_name == "ahrefs":
                result_data = await self._process_ahrefs_request(user_id, query_type, query_parameters, seed_keywords)
            elif tool_name == "ubersuggest":
                result_data = await self._process_ubersuggest_request(user_id, query_type, query_parameters, seed_keywords)
            else:
                raise ValueError(f"Unsupported tool: {tool_name}")
            
            # Update result with data
            external_result.raw_results = result_data.get("raw_data", {})
            external_result.processed_results = result_data.get("processed_data", {})
            external_result.keywords_data = result_data.get("keywords", [])
            external_result.clusters_data = result_data.get("clusters", [])
            external_result.total_keywords = len(result_data.get("keywords", []))
            external_result.total_clusters = len(result_data.get("clusters", []))
            external_result.status = "completed"
            external_result.is_processed = True
            external_result.processed_at = datetime.utcnow()
            external_result.processing_time_ms = result_data.get("processing_time_ms", 0)
            external_result.api_calls_made = result_data.get("api_calls_made", 0)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result_data)
            external_result.data_quality_score = quality_metrics["data_quality"]
            external_result.completeness_score = quality_metrics["completeness"]
            external_result.relevance_score = quality_metrics["relevance"]
            
            self.db.commit()
            
            logger.info("External tool request completed successfully", 
                       result_id=result_id,
                       keywords_count=external_result.total_keywords)
            
            return external_result.to_dict()
            
        except Exception as e:
            logger.error("Failed to process external tool request", 
                        error=str(e), 
                        user_id=user_id)
            
            # Update result with error
            if 'external_result' in locals():
                external_result.status = "failed"
                external_result.error_message = str(e)
                external_result.error_count = 1
                self.db.commit()
            
            raise
    
    async def _process_semrush_request(
        self, 
        user_id: str, 
        query_type: str, 
        query_parameters: Dict[str, Any], 
        seed_keywords: List[str]
    ) -> Dict[str, Any]:
        """Process Semrush API request"""
        try:
            # Get API key
            api_key_data = await self.api_key_service.get_api_key(user_id, ServiceName.SEMRUSH)
            if not api_key_data:
                raise ValueError("No Semrush API key found")
            
            api_key = api_key_data["api_key"]
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            start_time = datetime.utcnow()
            api_calls = 0
            keywords = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in seed_keywords:
                    try:
                        # Keyword overview
                        overview_data = await self._semrush_keyword_overview(session, api_key, keyword)
                        if overview_data:
                            keywords.extend(overview_data)
                            api_calls += 1
                        
                        # Related keywords
                        related_data = await self._semrush_related_keywords(session, api_key, keyword)
                        if related_data:
                            keywords.extend(related_data)
                            api_calls += 1
                        
                        # Keyword difficulty
                        difficulty_data = await self._semrush_keyword_difficulty(session, api_key, keyword)
                        if difficulty_data:
                            # Merge difficulty data with existing keywords
                            self._merge_keyword_data(keywords, difficulty_data)
                            api_calls += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process keyword {keyword} with Semrush", error=str(e))
                        continue
            
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Process and clean data
            processed_keywords = self._process_semrush_keywords(keywords)
            
            return {
                "raw_data": {"keywords": keywords},
                "processed_data": {"keywords": processed_keywords},
                "keywords": processed_keywords,
                "clusters": [],
                "processing_time_ms": processing_time,
                "api_calls_made": api_calls
            }
            
        except Exception as e:
            logger.error("Failed to process Semrush request", error=str(e))
            raise
    
    async def _semrush_keyword_overview(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get keyword overview from Semrush"""
        try:
            params = {
                'key': api_key,
                'type': 'phrase_organic',
                'phrase': keyword,
                'database': 'us',
                'export_columns': 'Ph,Po,Pp,Pd,Nq,Cp,Ur,Tr,Tc,Co,Nr,Td'
            }
            
            async with session.get(f"{self.semrush_url}/", params=params, headers=self.headers) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_semrush_csv(text)
                else:
                    logger.warning(f"Semrush API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Semrush keyword overview", error=str(e))
            return []
    
    async def _semrush_related_keywords(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get related keywords from Semrush"""
        try:
            params = {
                'key': api_key,
                'type': 'phrase_related',
                'phrase': keyword,
                'database': 'us',
                'export_columns': 'Ph,Po,Pp,Pd,Nq,Cp,Ur,Tr,Tc,Co,Nr,Td'
            }
            
            async with session.get(f"{self.semrush_url}/", params=params, headers=self.headers) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_semrush_csv(text)
                else:
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Semrush related keywords", error=str(e))
            return []
    
    async def _semrush_keyword_difficulty(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get keyword difficulty from Semrush"""
        try:
            params = {
                'key': api_key,
                'type': 'phrase_kdi',
                'phrase': keyword,
                'database': 'us',
                'export_columns': 'Ph,Kd'
            }
            
            async with session.get(f"{self.semrush_url}/", params=params, headers=self.headers) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_semrush_csv(text)
                else:
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Semrush keyword difficulty", error=str(e))
            return []
    
    def _parse_semrush_csv(self, csv_text: str) -> List[Dict[str, Any]]:
        """Parse Semrush CSV response"""
        try:
            import csv
            import io
            
            keywords = []
            reader = csv.DictReader(io.StringIO(csv_text))
            
            for row in reader:
                keyword_data = {
                    "keyword": row.get("Ph", ""),
                    "search_volume": int(row.get("Nq", 0)) if row.get("Nq") else 0,
                    "cpc": float(row.get("Cp", 0)) if row.get("Cp") else 0,
                    "competition": float(row.get("Co", 0)) if row.get("Co") else 0,
                    "difficulty": float(row.get("Kd", 0)) if row.get("Kd") else 0,
                    "trend": row.get("Tr", ""),
                    "source": "semrush"
                }
                
                if keyword_data["keyword"]:
                    keywords.append(keyword_data)
            
            return keywords
            
        except Exception as e:
            logger.error("Failed to parse Semrush CSV", error=str(e))
            return []
    
    def _process_semrush_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean Semrush keywords"""
        processed = []
        seen_keywords = set()
        
        for keyword in keywords:
            if not keyword.get("keyword") or keyword["keyword"] in seen_keywords:
                continue
            
            seen_keywords.add(keyword["keyword"])
            
            # Clean and standardize data
            processed_keyword = {
                "keyword": keyword["keyword"].strip(),
                "search_volume": keyword.get("search_volume", 0),
                "cpc": keyword.get("cpc", 0.0),
                "competition": keyword.get("competition", 0.0),
                "difficulty": keyword.get("difficulty", 0.0),
                "trend": keyword.get("trend", ""),
                "source": "semrush",
                "quality_score": self._calculate_keyword_quality(keyword)
            }
            
            processed.append(processed_keyword)
        
        return processed
    
    async def _process_ahrefs_request(
        self, 
        user_id: str, 
        query_type: str, 
        query_parameters: Dict[str, Any], 
        seed_keywords: List[str]
    ) -> Dict[str, Any]:
        """Process Ahrefs API request"""
        try:
            # Get API key
            api_key_data = await self.api_key_service.get_api_key(user_id, ServiceName.AHREFS)
            if not api_key_data:
                raise ValueError("No Ahrefs API key found")
            
            api_key = api_key_data["api_key"]
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            start_time = datetime.utcnow()
            api_calls = 0
            keywords = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in seed_keywords:
                    try:
                        # Keyword ideas
                        ideas_data = await self._ahrefs_keyword_ideas(session, api_key, keyword)
                        if ideas_data:
                            keywords.extend(ideas_data)
                            api_calls += 1
                        
                        # Related keywords
                        related_data = await self._ahrefs_related_keywords(session, api_key, keyword)
                        if related_data:
                            keywords.extend(related_data)
                            api_calls += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process keyword {keyword} with Ahrefs", error=str(e))
                        continue
            
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Process and clean data
            processed_keywords = self._process_ahrefs_keywords(keywords)
            
            return {
                "raw_data": {"keywords": keywords},
                "processed_data": {"keywords": processed_keywords},
                "keywords": processed_keywords,
                "clusters": [],
                "processing_time_ms": processing_time,
                "api_calls_made": api_calls
            }
            
        except Exception as e:
            logger.error("Failed to process Ahrefs request", error=str(e))
            raise
    
    async def _ahrefs_keyword_ideas(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get keyword ideas from Ahrefs"""
        try:
            headers = {
                **self.headers,
                'Authorization': f'Bearer {api_key}'
            }
            
            payload = {
                "target": keyword,
                "mode": "phrase",
                "limit": 100
            }
            
            async with session.post(
                f"{self.ahrefs_url}/keywords/ideas", 
                json=payload, 
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_ahrefs_keywords(data)
                else:
                    logger.warning(f"Ahrefs API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Ahrefs keyword ideas", error=str(e))
            return []
    
    async def _ahrefs_related_keywords(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get related keywords from Ahrefs"""
        try:
            headers = {
                **self.headers,
                'Authorization': f'Bearer {api_key}'
            }
            
            payload = {
                "target": keyword,
                "mode": "phrase",
                "limit": 100
            }
            
            async with session.post(
                f"{self.ahrefs_url}/keywords/related", 
                json=payload, 
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_ahrefs_keywords(data)
                else:
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Ahrefs related keywords", error=str(e))
            return []
    
    def _parse_ahrefs_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Ahrefs API response"""
        try:
            keywords = []
            
            if "keywords" in data:
                for item in data["keywords"]:
                    keyword_data = {
                        "keyword": item.get("keyword", ""),
                        "search_volume": item.get("search_volume", 0),
                        "cpc": item.get("cpc", 0.0),
                        "difficulty": item.get("keyword_difficulty", 0.0),
                        "source": "ahrefs"
                    }
                    
                    if keyword_data["keyword"]:
                        keywords.append(keyword_data)
            
            return keywords
            
        except Exception as e:
            logger.error("Failed to parse Ahrefs response", error=str(e))
            return []
    
    def _process_ahrefs_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean Ahrefs keywords"""
        processed = []
        seen_keywords = set()
        
        for keyword in keywords:
            if not keyword.get("keyword") or keyword["keyword"] in seen_keywords:
                continue
            
            seen_keywords.add(keyword["keyword"])
            
            processed_keyword = {
                "keyword": keyword["keyword"].strip(),
                "search_volume": keyword.get("search_volume", 0),
                "cpc": keyword.get("cpc", 0.0),
                "difficulty": keyword.get("difficulty", 0.0),
                "source": "ahrefs",
                "quality_score": self._calculate_keyword_quality(keyword)
            }
            
            processed.append(processed_keyword)
        
        return processed
    
    async def _process_ubersuggest_request(
        self, 
        user_id: str, 
        query_type: str, 
        query_parameters: Dict[str, Any], 
        seed_keywords: List[str]
    ) -> Dict[str, Any]:
        """Process Ubersuggest API request"""
        try:
            # Get API key
            api_key_data = await self.api_key_service.get_api_key(user_id, ServiceName.UBERSUGGEST)
            if not api_key_data:
                raise ValueError("No Ubersuggest API key found")
            
            api_key = api_key_data["api_key"]
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            start_time = datetime.utcnow()
            api_calls = 0
            keywords = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in seed_keywords:
                    try:
                        # Keyword suggestions
                        suggestions_data = await self._ubersuggest_keyword_suggestions(session, api_key, keyword)
                        if suggestions_data:
                            keywords.extend(suggestions_data)
                            api_calls += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process keyword {keyword} with Ubersuggest", error=str(e))
                        continue
            
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Process and clean data
            processed_keywords = self._process_ubersuggest_keywords(keywords)
            
            return {
                "raw_data": {"keywords": keywords},
                "processed_data": {"keywords": processed_keywords},
                "keywords": processed_keywords,
                "clusters": [],
                "processing_time_ms": processing_time,
                "api_calls_made": api_calls
            }
            
        except Exception as e:
            logger.error("Failed to process Ubersuggest request", error=str(e))
            raise
    
    async def _ubersuggest_keyword_suggestions(self, session: aiohttp.ClientSession, api_key: str, keyword: str) -> List[Dict[str, Any]]:
        """Get keyword suggestions from Ubersuggest"""
        try:
            headers = {
                **self.headers,
                'Authorization': f'Bearer {api_key}'
            }
            
            params = {
                'keyword': keyword,
                'country': 'us',
                'language': 'en'
            }
            
            async with session.get(
                f"{self.ubersuggest_url}/keyword_suggestions", 
                params=params, 
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_ubersuggest_keywords(data)
                else:
                    logger.warning(f"Ubersuggest API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error("Failed to get Ubersuggest keyword suggestions", error=str(e))
            return []
    
    def _parse_ubersuggest_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Ubersuggest API response"""
        try:
            keywords = []
            
            if "data" in data and "keywords" in data["data"]:
                for item in data["data"]["keywords"]:
                    keyword_data = {
                        "keyword": item.get("keyword", ""),
                        "search_volume": item.get("search_volume", 0),
                        "cpc": item.get("cpc", 0.0),
                        "difficulty": item.get("difficulty", 0.0),
                        "source": "ubersuggest"
                    }
                    
                    if keyword_data["keyword"]:
                        keywords.append(keyword_data)
            
            return keywords
            
        except Exception as e:
            logger.error("Failed to parse Ubersuggest response", error=str(e))
            return []
    
    def _process_ubersuggest_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean Ubersuggest keywords"""
        processed = []
        seen_keywords = set()
        
        for keyword in keywords:
            if not keyword.get("keyword") or keyword["keyword"] in seen_keywords:
                continue
            
            seen_keywords.add(keyword["keyword"])
            
            processed_keyword = {
                "keyword": keyword["keyword"].strip(),
                "search_volume": keyword.get("search_volume", 0),
                "cpc": keyword.get("cpc", 0.0),
                "difficulty": keyword.get("difficulty", 0.0),
                "source": "ubersuggest",
                "quality_score": self._calculate_keyword_quality(keyword)
            }
            
            processed.append(processed_keyword)
        
        return processed
    
    def _calculate_keyword_quality(self, keyword: Dict[str, Any]) -> float:
        """Calculate keyword quality score"""
        try:
            volume = keyword.get("search_volume", 0)
            difficulty = keyword.get("difficulty", 100)
            cpc = keyword.get("cpc", 0)
            
            # Volume score (0-40 points)
            volume_score = min(40, volume / 1000 * 4)
            
            # Difficulty score (0-30 points, lower difficulty is better)
            difficulty_score = max(0, 30 - (difficulty / 100 * 30))
            
            # CPC score (0-30 points, higher CPC is better)
            cpc_score = min(30, cpc * 10)
            
            total_score = volume_score + difficulty_score + cpc_score
            return min(100, max(0, total_score))
            
        except Exception:
            return 0.0
    
    def _merge_keyword_data(self, keywords: List[Dict[str, Any]], new_data: List[Dict[str, Any]]) -> None:
        """Merge new keyword data with existing keywords"""
        try:
            keyword_map = {kw["keyword"]: kw for kw in keywords}
            
            for new_kw in new_data:
                keyword = new_kw["keyword"]
                if keyword in keyword_map:
                    # Update existing keyword
                    keyword_map[keyword].update(new_kw)
                else:
                    # Add new keyword
                    keywords.append(new_kw)
                    
        except Exception as e:
            logger.error("Failed to merge keyword data", error=str(e))
    
    def _calculate_quality_metrics(self, result_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for external tool results"""
        try:
            keywords = result_data.get("keywords", [])
            
            if not keywords:
                return {"data_quality": 0.0, "completeness": 0.0, "relevance": 0.0}
            
            # Data quality (based on completeness of fields)
            quality_scores = []
            for kw in keywords:
                score = 0
                if kw.get("search_volume", 0) > 0:
                    score += 1
                if kw.get("difficulty", 0) > 0:
                    score += 1
                if kw.get("cpc", 0) > 0:
                    score += 1
                if kw.get("keyword"):
                    score += 1
                quality_scores.append(score / 4 * 100)
            
            data_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Completeness (based on number of keywords vs expected)
            expected_keywords = 50  # Expected minimum
            completeness = min(100, (len(keywords) / expected_keywords) * 100)
            
            # Relevance (based on quality scores)
            relevance_scores = [kw.get("quality_score", 0) for kw in keywords if "quality_score" in kw]
            relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            return {
                "data_quality": round(data_quality, 2),
                "completeness": round(completeness, 2),
                "relevance": round(relevance, 2)
            }
            
        except Exception as e:
            logger.error("Failed to calculate quality metrics", error=str(e))
            return {"data_quality": 0.0, "completeness": 0.0, "relevance": 0.0}
    
    async def get_external_tool_results(
        self,
        user_id: str,
        workflow_session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get external tool results with optional filtering"""
        try:
            query = self.db.query(ExternalToolResult).filter(ExternalToolResult.user_id == user_id)
            
            if workflow_session_id:
                query = query.filter(ExternalToolResult.workflow_session_id == workflow_session_id)
            
            if tool_name:
                query = query.filter(ExternalToolResult.tool_name == tool_name)
            
            if status:
                query = query.filter(ExternalToolResult.status == status)
            
            results = query.order_by(ExternalToolResult.created_at.desc()).offset(offset).limit(limit).all()
            
            return [result.to_dict() for result in results]
            
        except Exception as e:
            logger.error("Failed to get external tool results", error=str(e))
            raise
    
    async def retry_failed_request(self, result_id: str, user_id: str) -> Dict[str, Any]:
        """Retry a failed external tool request"""
        try:
            result = self.db.query(ExternalToolResult).filter(
                ExternalToolResult.id == result_id,
                ExternalToolResult.user_id == user_id
            ).first()
            
            if not result:
                raise ValueError("External tool result not found")
            
            if result.status != "failed":
                raise ValueError("Can only retry failed requests")
            
            if result.retry_count >= 3:
                raise ValueError("Maximum retry attempts reached")
            
            # Update retry count
            result.retry_count += 1
            result.last_retry_at = datetime.utcnow()
            result.status = "pending"
            result.error_message = None
            self.db.commit()
            
            # Retry the request
            return await self.process_external_tool_request(
                user_id=result.user_id,
                workflow_session_id=result.workflow_session_id,
                tool_name=result.tool_name,
                query_type=result.query_type,
                query_parameters=result.query_parameters,
                seed_keywords=result.seed_keywords,
                trend_analysis_id=result.trend_analysis_id
            )
            
        except Exception as e:
            logger.error("Failed to retry external tool request", error=str(e))
            raise
