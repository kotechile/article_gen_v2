"""
KeywordService for CSV processing + DataForSEO
"""

import asyncio
import aiohttp
import csv
import io
import json
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.keyword_data import KeywordData, KeywordStatus, KeywordSource

logger = structlog.get_logger()
settings = get_settings()

class KeywordService:
    """Service for keyword data management and processing"""
    
    def __init__(self):
        self.dataforseo_username = settings.dataforseo_username
        self.dataforseo_password = settings.dataforseo_password
        self.max_file_size = settings.max_file_size
        self.allowed_file_types = settings.allowed_file_types
        
        # DataForSEO API configuration
        self.dataforseo_base_url = "https://api.dataforseo.com/v3"
        self.dataforseo_cost_per_keyword = 0.0008  # $0.0008 per keyword
    
    async def generate_seed_keywords(
        self, 
        search_term: str, 
        selected_trends: List[str], 
        content_ideas: List[str],
        user_id: int
    ) -> Dict[str, Any]:
        """
        Generate seed keywords based on search term, trends, and content ideas
        """
        try:
            logger.info("Generating seed keywords", 
                       search_term=search_term, 
                       trends_count=len(selected_trends),
                       content_ideas_count=len(content_ideas))
            
            # Generate keywords using LLM
            seed_keywords = await self._generate_keywords_with_llm(
                search_term, selected_trends, content_ideas
            )
            
            # Categorize keywords
            categorized_keywords = self._categorize_keywords(seed_keywords)
            
            # Generate export formats for external tools
            export_formats = self._generate_export_formats(categorized_keywords)
            
            return {
                "search_term": search_term,
                "total_keywords": len(seed_keywords),
                "categorized_keywords": categorized_keywords,
                "export_formats": export_formats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to generate seed keywords", error=str(e))
            raise

    async def process_external_keyword_data(
        self,
        df: pd.DataFrame,
        tool_name: str,
        search_term: str,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Process keyword data from external tools (Ahrefs, Semrush, etc.)
        """
        try:
            logger.info("Processing external keyword data",
                       tool_name=tool_name,
                       search_term=search_term,
                       rows=len(df))
            
            # Normalize data based on tool
            normalized_df = self._normalize_external_data(df, tool_name)
            
            # Analyze keyword clusters
            clusters = self._analyze_keyword_clusters(normalized_df)
            
            # Generate content strategy
            content_strategy = self._generate_content_strategy(clusters, search_term)
            
            return {
                "tool_name": tool_name,
                "search_term": search_term,
                "keywords_processed": len(normalized_df),
                "clusters_found": len(clusters),
                "content_strategy": content_strategy,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to process external keyword data", error=str(e))
            raise

    async def upload_csv(self, user_id: int, file: BinaryIO, filename: str, 
                        trend_analysis_id: Optional[int] = None) -> Dict[str, Any]:
        """Upload and process CSV file with keywords"""
        try:
            # Validate file
            self._validate_file(file, filename)
            
            # Create keyword data record
            db = next(get_db())
            keyword_data = KeywordData(
                user_id=user_id,
                trend_analysis_id=trend_analysis_id,
                status=KeywordStatus.PROCESSING,
                source=KeywordSource.CSV_UPLOAD,
                original_filename=filename,
                file_size=len(file.read())
            )
            file.seek(0)  # Reset file pointer
            db.add(keyword_data)
            db.commit()
            db.refresh(keyword_data)
            
            # Start background processing
            asyncio.create_task(self._process_csv_file(keyword_data.id, file))
            
            logger.info("CSV upload initiated", keyword_data_id=keyword_data.id, filename=filename)
            return keyword_data.to_dict()
            
        except Exception as e:
            logger.error("Failed to upload CSV", error=str(e))
            raise
    
    async def crawl_keywords(self, user_id: int, keywords: List[str], 
                           trend_analysis_id: Optional[int] = None) -> Dict[str, Any]:
        """Crawl keywords using DataForSEO"""
        try:
            # Validate keywords
            if not keywords or len(keywords) > 1000:
                raise ValueError("Keywords list must contain 1-1000 keywords")
            
            # Calculate estimated cost
            estimated_cost = len(keywords) * self.dataforseo_cost_per_keyword
            
            # Create keyword data record
            db = next(get_db())
            keyword_data = KeywordData(
                user_id=user_id,
                trend_analysis_id=trend_analysis_id,
                status=KeywordStatus.PROCESSING,
                source=KeywordSource.DATAFORSEO,
                estimated_cost=estimated_cost
            )
            db.add(keyword_data)
            db.commit()
            db.refresh(keyword_data)
            
            # Start background crawling
            asyncio.create_task(self._crawl_keywords_data(keyword_data.id, keywords))
            
            logger.info("Keyword crawl initiated", keyword_data_id=keyword_data.id, keyword_count=len(keywords))
            return keyword_data.to_dict()
            
        except Exception as e:
            logger.error("Failed to initiate keyword crawl", error=str(e))
            raise
    
    async def get_keyword_data(self, keyword_data_id: int) -> Dict[str, Any]:
        """Get keyword data by ID"""
        try:
            db = next(get_db())
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not keyword_data:
                raise ValueError("Keyword data not found")
            
            return keyword_data.to_dict()
            
        except Exception as e:
            logger.error("Failed to get keyword data", keyword_data_id=keyword_data_id, error=str(e))
            raise
    
    async def _process_csv_file(self, keyword_data_id: int, file: BinaryIO):
        """Process CSV file in background"""
        try:
            db = next(get_db())
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not keyword_data:
                return
            
            # Parse CSV file
            file_content = file.read().decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(file_content))
            
            keywords = []
            for row in csv_reader:
                keyword = self._parse_csv_row(row)
                if keyword:
                    keywords.append(keyword)
            
            # Process keywords with additional data
            processed_keywords = await self._enrich_keywords(keywords)
            
            # Calculate priority scores
            for keyword in processed_keywords:
                keyword["priority_score"] = self._calculate_priority_score(keyword)
            
            # Update keyword data
            keyword_data.mark_completed(processed_keywords)
            keyword_data.actual_cost = 0  # CSV upload is free
            db.commit()
            
            logger.info("CSV processing completed", keyword_data_id=keyword_data_id, keyword_count=len(processed_keywords))
            
        except Exception as e:
            logger.error("CSV processing failed", keyword_data_id=keyword_data_id, error=str(e))
            
            # Mark as failed
            try:
                db = next(get_db())
                keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
                if keyword_data:
                    keyword_data.mark_failed(str(e))
                    db.commit()
            except:
                pass
    
    async def _crawl_keywords_data(self, keyword_data_id: int, keywords: List[str]):
        """Crawl keywords using DataForSEO in background"""
        try:
            db = next(get_db())
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not keyword_data:
                return
            
            # Check if DataForSEO is available
            if not self.dataforseo_username or not self.dataforseo_password:
                # Use mock data if DataForSEO not available
                processed_keywords = await self._get_mock_keyword_data(keywords)
            else:
                # Call DataForSEO API
                processed_keywords = await self._call_dataforseo_api(keywords)
            
            # Calculate priority scores
            for keyword in processed_keywords:
                keyword["priority_score"] = self._calculate_priority_score(keyword)
            
            # Update keyword data
            keyword_data.mark_completed(processed_keywords)
            keyword_data.actual_cost = len(keywords) * self.dataforseo_cost_per_keyword
            db.commit()
            
            logger.info("Keyword crawling completed", keyword_data_id=keyword_data_id, keyword_count=len(processed_keywords))
            
        except Exception as e:
            logger.error("Keyword crawling failed", keyword_data_id=keyword_data_id, error=str(e))
            
            # Mark as failed
            try:
                db = next(get_db())
                keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
                if keyword_data:
                    keyword_data.mark_failed(str(e))
                    db.commit()
            except:
                pass
    
    def _validate_file(self, file: BinaryIO, filename: str):
        """Validate uploaded file"""
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            raise ValueError(f"File too large. Maximum size: {self.max_file_size} bytes")
        
        if file_size == 0:
            raise ValueError("File is empty")
        
        # Check file type
        if not filename.lower().endswith('.csv'):
            raise ValueError("Only CSV files are allowed")
        
        # Check content type
        file_content = file.read(1024).decode('utf-8', errors='ignore')
        file.seek(0)  # Reset to beginning
        
        if 'keyword' not in file_content.lower():
            raise ValueError("CSV file must contain 'keyword' column")
    
    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse CSV row into keyword object"""
        try:
            keyword = row.get('keyword', '').strip()
            if not keyword:
                return None
            
            # Parse numeric fields
            search_volume = int(row.get('search_volume', 0)) if row.get('search_volume') else 0
            difficulty = int(row.get('difficulty', 50)) if row.get('difficulty') else 50
            cpc = float(row.get('cpc', 0)) if row.get('cpc') else 0
            
            # Parse string fields
            competition = row.get('competition', 'medium').lower()
            intent = row.get('intent', 'informational').lower()
            
            # Validate competition
            if competition not in ['low', 'medium', 'high']:
                competition = 'medium'
            
            # Validate intent
            if intent not in ['informational', 'commercial', 'navigational', 'transactional']:
                intent = 'informational'
            
            return {
                "keyword": keyword,
                "search_volume": search_volume,
                "difficulty": difficulty,
                "cpc": cpc,
                "competition": competition,
                "intent": intent,
                "priority_score": 0.0  # Will be calculated later
            }
            
        except Exception as e:
            logger.warning("Failed to parse CSV row", row=row, error=str(e))
            return None
    
    async def _enrich_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich keywords with additional data"""
        enriched_keywords = []
        
        for keyword in keywords:
            # Add SERP analysis
            keyword["serp_analysis"] = self._generate_serp_analysis(keyword)
            
            # Add NLP terms
            keyword["nlp_terms"] = self._extract_nlp_terms(keyword["keyword"])
            
            # Add People Also Ask
            keyword["people_also_ask"] = self._generate_people_also_ask(keyword["keyword"])
            
            # Add internal link suggestions
            keyword["internal_link_suggestions"] = self._generate_internal_links(keyword["keyword"])
            
            enriched_keywords.append(keyword)
        
        return enriched_keywords
    
    def _generate_serp_analysis(self, keyword: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock SERP analysis"""
        # Mock implementation - replace with actual SERP analysis
        return {
            "top_10_avg_domain_authority": 45.0 + (hash(keyword["keyword"]) % 30),
            "top_10_avg_page_authority": 40.0 + (hash(keyword["keyword"]) % 25),
            "top_10_avg_content_length": 2000 + (hash(keyword["keyword"]) % 1000),
            "top_10_avg_backlinks": 500 + (hash(keyword["keyword"]) % 500),
            "serp_weakness_score": 0.3 + (hash(keyword["keyword"]) % 50) / 100
        }
    
    def _extract_nlp_terms(self, keyword: str) -> List[str]:
        """Extract NLP terms from keyword"""
        # Simple keyword splitting - replace with actual NLP processing
        terms = keyword.lower().split()
        return [term for term in terms if len(term) > 2]
    
    def _generate_people_also_ask(self, keyword: str) -> List[str]:
        """Generate People Also Ask questions"""
        # Mock implementation - replace with actual PAA extraction
        return [
            f"What is {keyword}?",
            f"How to use {keyword}?",
            f"Best {keyword} guide",
            f"{keyword} vs alternatives"
        ]
    
    def _generate_internal_links(self, keyword: str) -> List[str]:
        """Generate internal link suggestions"""
        # Mock implementation - replace with actual internal linking analysis
        return [
            f"{keyword} guide",
            f"{keyword} tips",
            f"{keyword} examples",
            f"related to {keyword}"
        ]
    
    async def _call_dataforseo_api(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Call DataForSEO API using real integration"""
        try:
            # Import the real DataForSEO integration
            from ..integrations.dataforseo import dataforseo_api
            
            # Process keywords in batches to avoid rate limits
            batch_size = 10
            all_results = []
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i + batch_size]
                
                # Get keyword ideas for each keyword in the batch
                for keyword in batch:
                    try:
                        keyword_ideas = await dataforseo_api.get_keyword_ideas(
                            seed_keyword=keyword,
                            language_code="en",
                            location_code=2840,  # US
                            limit=5
                        )
                        all_results.extend(keyword_ideas)
                        
                        # Add small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"DataForSEO API error for keyword {keyword}: {e}")
                        # Add mock data for this keyword
                        all_results.append({
                            "keyword": keyword,
                            "search_volume": 100,
                            "difficulty": 30,
                            "cpc": 1.0,
                            "competition": "medium",
                            "intent": "informational",
                            "priority_score": 0.5,
                            "serp_analysis": {
                                "top_10_avg_domain_authority": 50,
                                "top_10_avg_page_authority": 40,
                                "top_10_avg_content_length": 2000,
                                "featured_snippets": 0,
                                "local_pack": False
                            },
                            "related_keywords": [],
                            "data_source": "dataforseo_fallback"
                        })
            
            return all_results
            
        except Exception as e:
            logger.error("DataForSEO API call failed", error=str(e))
            return await self._get_mock_keyword_data(keywords)
    
    async def _get_mock_keyword_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Get mock keyword data"""
        mock_keywords = []
        
        for keyword in keywords:
            # Generate mock data based on keyword
            hash_val = hash(keyword)
            
            mock_keyword = {
                "keyword": keyword,
                "search_volume": 100 + (hash_val % 2000),
                "difficulty": 20 + (hash_val % 60),
                "cpc": 0.5 + (hash_val % 5),
                "competition": ["low", "medium", "high"][hash_val % 3],
                "intent": ["informational", "commercial", "navigational", "transactional"][hash_val % 4],
                "priority_score": 0.0,  # Will be calculated
                "serp_analysis": {
                    "top_10_avg_domain_authority": 30 + (hash_val % 40),
                    "top_10_avg_page_authority": 25 + (hash_val % 35),
                    "top_10_avg_content_length": 1500 + (hash_val % 1500),
                    "top_10_avg_backlinks": 200 + (hash_val % 800),
                    "serp_weakness_score": 0.2 + (hash_val % 60) / 100
                },
                "nlp_terms": keyword.lower().split(),
                "people_also_ask": [
                    f"What is {keyword}?",
                    f"How to {keyword}?",
                    f"Best {keyword} guide"
                ],
                "internal_link_suggestions": [
                    f"{keyword} guide",
                    f"{keyword} tips",
                    f"related to {keyword}"
                ]
            }
            
            mock_keywords.append(mock_keyword)
        
        return mock_keywords
    
    def _calculate_priority_score(self, keyword: Dict[str, Any]) -> float:
        """Calculate priority score for keyword"""
        search_volume = keyword.get("search_volume", 0)
        difficulty = keyword.get("difficulty", 50)
        cpc = keyword.get("cpc", 0)
        serp_weakness = keyword.get("serp_analysis", {}).get("serp_weakness_score", 0.5)
        
        # Normalize search volume (0-1 scale)
        volume_score = min(search_volume / 10000, 1.0)
        
        # Normalize difficulty (invert so lower difficulty = higher score)
        difficulty_score = 1.0 - (difficulty / 100)
        
        # Normalize CPC (0-1 scale)
        cpc_score = min(cpc / 10, 1.0)
        
        # SERP weakness (higher weakness = higher opportunity)
        serp_score = serp_weakness
        
        # Weighted average
        priority_score = (
            volume_score * 0.3 +
            difficulty_score * 0.3 +
            cpc_score * 0.2 +
            serp_score * 0.2
        )
        
        return min(max(priority_score, 0), 1)
    
    async def get_user_keyword_data(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's keyword data"""
        try:
            db = next(get_db())
            keyword_data = db.query(KeywordData).filter(
                KeywordData.user_id == user_id
            ).order_by(KeywordData.created_at.desc()).offset(offset).limit(limit).all()
            
            return [kd.to_dict() for kd in keyword_data]
            
        except Exception as e:
            logger.error("Failed to get user keyword data", user_id=user_id, error=str(e))
            raise
    
    async def delete_keyword_data(self, keyword_data_id: int, user_id: int) -> bool:
        """Delete keyword data"""
        try:
            db = next(get_db())
            keyword_data = db.get_KeywordData_by_id(
                KeywordData.id == keyword_data_id,
                KeywordData.user_id == user_id
            )
            
            if not keyword_data:
                raise ValueError("Keyword data not found")
            
            db.delete(keyword_data)
            db.commit()
            
            logger.info("Keyword data deleted", keyword_data_id=keyword_data_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete keyword data", keyword_data_id=keyword_data_id, error=str(e))
            raise
    
    async def get_keyword_analytics(self, keyword_data_id: int) -> Dict[str, Any]:
        """Get keyword analytics and insights"""
        try:
            db = next(get_db())
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not keyword_data or not keyword_data.keywords:
                raise ValueError("Keyword data not found or empty")
            
            # Calculate analytics
            analytics = {
                "total_keywords": len(keyword_data.keywords),
                "serp_analysis_summary": keyword_data.get_serp_analysis_summary(),
                "intent_distribution": keyword_data.get_intent_distribution(),
                "difficulty_distribution": keyword_data.get_difficulty_distribution(),
                "top_keywords": keyword_data.get_top_keywords(10),
                "high_priority_keywords": keyword_data.get_high_priority_keywords(0.7),
                "average_priority_score": sum(k.get("priority_score", 0) for k in keyword_data.keywords) / len(keyword_data.keywords)
            }
            
            return analytics
            
        except Exception as e:
            logger.error("Failed to get keyword analytics", keyword_data_id=keyword_data_id, error=str(e))
            raise
    
    async def analyze_bulk_keywords(self, keywords: List[str], user_id: int) -> Dict[str, Any]:
        """Analyze bulk keywords using DataForSEO API"""
        try:
            # Import DataForSEO integration
            from ..integrations.dataforseo import dataforseo_api
            
            # Check if DataForSEO is available
            if not self.dataforseo_username or not self.dataforseo_password:
                logger.warning("DataForSEO not configured, using mock data")
                return {
                    "keywords": await self._get_mock_keyword_data(keywords),
                    "total_cost": 0.0,
                    "data_source": "mock"
                }
            
            # Calculate estimated cost
            estimated_cost = len(keywords) * self.dataforseo_cost_per_keyword
            
            logger.info(f"Starting bulk keyword analysis", 
                       keyword_count=len(keywords), 
                       estimated_cost=estimated_cost)
            
            # Process keywords in batches
            batch_size = 20
            all_results = []
            total_cost = 0.0
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i + batch_size]
                
                try:
                    # Use DataForSEO bulk keyword analysis
                    batch_results = await dataforseo_api.get_bulk_keyword_analysis(
                        keywords=batch,
                        language_code="en",
                        location_code=2840
                    )
                    
                    all_results.extend(batch_results)
                    total_cost += len(batch) * self.dataforseo_cost_per_keyword
                    
                    # Add delay between batches to avoid rate limiting
                    if i + batch_size < len(keywords):
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    logger.error(f"DataForSEO batch error: {e}")
                    # Fallback to individual keyword analysis
                    for keyword in batch:
                        try:
                            keyword_data = await dataforseo_api.get_keyword_ideas(
                                seed_keyword=keyword,
                                language_code="en",
                                location_code=2840,
                                limit=1
                            )
                            if keyword_data:
                                all_results.extend(keyword_data)
                                total_cost += self.dataforseo_cost_per_keyword
                        except Exception as keyword_error:
                            logger.error(f"DataForSEO individual keyword error for {keyword}: {keyword_error}")
                            # Add fallback data
                            all_results.append({
                                "keyword": keyword,
                                "search_volume": 100,
                                "difficulty": 30,
                                "cpc": 1.0,
                                "competition": "medium",
                                "intent": "informational",
                                "priority_score": 0.5,
                                "data_source": "dataforseo_fallback"
                            })
            
            logger.info(f"Bulk keyword analysis completed", 
                       total_keywords=len(all_results), 
                       total_cost=total_cost)
            
            return {
                "keywords": all_results,
                "total_cost": total_cost,
                "data_source": "dataforseo"
            }
            
        except Exception as e:
            logger.error("Bulk keyword analysis failed", error=str(e))
            # Return mock data as fallback
            return {
                "keywords": await self._get_mock_keyword_data(keywords),
                "total_cost": 0.0,
                "data_source": "mock_fallback"
            }
