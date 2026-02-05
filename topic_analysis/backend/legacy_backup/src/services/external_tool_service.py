"""
External Tool Integration Service for enhanced research workflow
"""

import pandas as pd
import io
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from ..models.keyword_data import KeywordData, KeywordSource, KeywordStatus
from ..core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class ExternalToolService:
    """Service for integrating with external keyword research tools"""
    
    def __init__(self, db: Session):
        self.db = db
        self.max_file_size = settings.max_file_size
        self.allowed_file_types = settings.allowed_file_types
    
    async def process_external_tool_results(
        self, 
        file_content: bytes, 
        tool_type: str, 
        column_mapping: Dict[str, str],
        user_id: str,
        keyword_data_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process results from external keyword research tools"""
        try:
            logger.info("Processing external tool results", 
                       user_id=user_id, 
                       tool_type=tool_type,
                       file_size=len(file_content))
            
            # Validate file size
            if len(file_content) > self.max_file_size:
                raise ValueError(f"File size exceeds maximum limit of {self.max_file_size} bytes")
            
            # Read CSV file
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Process based on tool type
            if tool_type == "ahrefs":
                processed_data = await self._process_ahrefs_data(df, column_mapping)
            elif tool_type == "semrush":
                processed_data = await self._process_semrush_data(df, column_mapping)
            elif tool_type == "ubersuggest":
                processed_data = await self._process_ubersuggest_data(df, column_mapping)
            else:
                raise ValueError(f"Unsupported tool type: {tool_type}")
            
            # Save to database
            if keyword_data_id:
                # Update existing keyword data
                keyword_data = self.db.query(KeywordData).filter(
                    KeywordData.id == keyword_data_id,
                    KeywordData.user_id == user_id
                ).first()
                
                if keyword_data:
                    keyword_data.external_tool_source = KeywordSource(tool_type)
                    keyword_data.external_tool_metrics = processed_data["metrics"]
                    keyword_data.keywords = processed_data["keywords"]
                    keyword_data.keyword_count = len(processed_data["keywords"])
                    keyword_data.status = KeywordStatus.COMPLETED
                    keyword_data.completed_at = datetime.utcnow()
                else:
                    raise ValueError("Keyword data not found")
            else:
                # Create new keyword data
                keyword_data = KeywordData(
                    user_id=user_id,
                    status=KeywordStatus.COMPLETED,
                    source=KeywordSource.MANUAL,
                    external_tool_source=KeywordSource(tool_type),
                    external_tool_metrics=processed_data["metrics"],
                    keywords=processed_data["keywords"],
                    keyword_count=len(processed_data["keywords"]),
                    completed_at=datetime.utcnow()
                )
                
                self.db.add(keyword_data)
            
            self.db.commit()
            self.db.refresh(keyword_data)
            
            logger.info("External tool results processed successfully", 
                       user_id=user_id, 
                       tool_type=tool_type,
                       keywords_count=len(processed_data["keywords"]))
            
            return {
                "upload_id": str(uuid.uuid4()),
                "tool_type": tool_type,
                "keywords_processed": len(processed_data["keywords"]),
                "processing_status": "completed",
                "message": f"{tool_type.title()} results processed successfully",
                "keyword_data_id": keyword_data.id
            }
            
        except Exception as e:
            logger.error("Failed to process external tool results", error=str(e), user_id=user_id)
            self.db.rollback()
            raise
    
    async def _process_ahrefs_data(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process Ahrefs CSV data"""
        keywords = []
        metrics = {
            "total_keywords": len(df),
            "avg_volume": 0,
            "avg_difficulty": 0,
            "avg_cpc": 0
        }
        
        volume_sum = 0
        difficulty_sum = 0
        cpc_sum = 0
        valid_count = 0
        
        for index, row in df.iterrows():
            try:
                keyword_data = {
                    "keyword": str(row[column_mapping.get("keyword", "Keyword")]).strip(),
                    "search_volume": self._safe_int(row[column_mapping.get("volume", "Search Volume")]),
                    "difficulty": self._safe_float(row[column_mapping.get("difficulty", "KD")]),
                    "cpc": self._safe_float(row[column_mapping.get("cpc", "CPC")]),
                    "competition": str(row[column_mapping.get("competition", "Competition")]).strip(),
                    "intent": self._determine_intent(str(row[column_mapping.get("keyword", "Keyword")])),
                    "priority_score": 0.0,
                    "source": "ahrefs"
                }
                
                # Calculate priority score
                keyword_data["priority_score"] = self._calculate_priority_score(keyword_data)
                
                keywords.append(keyword_data)
                
                # Update metrics
                if keyword_data["search_volume"] > 0:
                    volume_sum += keyword_data["search_volume"]
                    valid_count += 1
                
                difficulty_sum += keyword_data["difficulty"]
                cpc_sum += keyword_data["cpc"]
                
            except Exception as e:
                logger.warning(f"Error processing Ahrefs row {index}: {str(e)}")
                continue
        
        # Calculate averages
        if valid_count > 0:
            metrics["avg_volume"] = volume_sum / valid_count
        metrics["avg_difficulty"] = difficulty_sum / len(keywords) if keywords else 0
        metrics["avg_cpc"] = cpc_sum / len(keywords) if keywords else 0
        
        return {
            "keywords": keywords,
            "metrics": metrics
        }
    
    async def _process_semrush_data(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process Semrush CSV data"""
        keywords = []
        metrics = {
            "total_keywords": len(df),
            "avg_volume": 0,
            "avg_difficulty": 0,
            "avg_cpc": 0
        }
        
        volume_sum = 0
        difficulty_sum = 0
        cpc_sum = 0
        valid_count = 0
        
        for index, row in df.iterrows():
            try:
                keyword_data = {
                    "keyword": str(row[column_mapping.get("keyword", "Keyword")]).strip(),
                    "search_volume": self._safe_int(row[column_mapping.get("volume", "Search Volume")]),
                    "difficulty": self._safe_float(row[column_mapping.get("difficulty", "KD")]),
                    "cpc": self._safe_float(row[column_mapping.get("cpc", "CPC")]),
                    "competition": str(row[column_mapping.get("competition", "Competition")]).strip(),
                    "intent": self._determine_intent(str(row[column_mapping.get("keyword", "Keyword")])),
                    "priority_score": 0.0,
                    "source": "semrush"
                }
                
                # Calculate priority score
                keyword_data["priority_score"] = self._calculate_priority_score(keyword_data)
                
                keywords.append(keyword_data)
                
                # Update metrics
                if keyword_data["search_volume"] > 0:
                    volume_sum += keyword_data["search_volume"]
                    valid_count += 1
                
                difficulty_sum += keyword_data["difficulty"]
                cpc_sum += keyword_data["cpc"]
                
            except Exception as e:
                logger.warning(f"Error processing Semrush row {index}: {str(e)}")
                continue
        
        # Calculate averages
        if valid_count > 0:
            metrics["avg_volume"] = volume_sum / valid_count
        metrics["avg_difficulty"] = difficulty_sum / len(keywords) if keywords else 0
        metrics["avg_cpc"] = cpc_sum / len(keywords) if keywords else 0
        
        return {
            "keywords": keywords,
            "metrics": metrics
        }
    
    async def _process_ubersuggest_data(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process Ubersuggest CSV data"""
        keywords = []
        metrics = {
            "total_keywords": len(df),
            "avg_volume": 0,
            "avg_difficulty": 0,
            "avg_cpc": 0
        }
        
        volume_sum = 0
        difficulty_sum = 0
        cpc_sum = 0
        valid_count = 0
        
        for index, row in df.iterrows():
            try:
                keyword_data = {
                    "keyword": str(row[column_mapping.get("keyword", "Keyword")]).strip(),
                    "search_volume": self._safe_int(row[column_mapping.get("volume", "Search Volume")]),
                    "difficulty": self._safe_float(row[column_mapping.get("difficulty", "SEO Difficulty")]),
                    "cpc": self._safe_float(row[column_mapping.get("cpc", "CPC")]),
                    "competition": str(row[column_mapping.get("competition", "Competition")]).strip(),
                    "intent": self._determine_intent(str(row[column_mapping.get("keyword", "Keyword")])),
                    "priority_score": 0.0,
                    "source": "ubersuggest"
                }
                
                # Calculate priority score
                keyword_data["priority_score"] = self._calculate_priority_score(keyword_data)
                
                keywords.append(keyword_data)
                
                # Update metrics
                if keyword_data["search_volume"] > 0:
                    volume_sum += keyword_data["search_volume"]
                    valid_count += 1
                
                difficulty_sum += keyword_data["difficulty"]
                cpc_sum += keyword_data["cpc"]
                
            except Exception as e:
                logger.warning(f"Error processing Ubersuggest row {index}: {str(e)}")
                continue
        
        # Calculate averages
        if valid_count > 0:
            metrics["avg_volume"] = volume_sum / valid_count
        metrics["avg_difficulty"] = difficulty_sum / len(keywords) if keywords else 0
        metrics["avg_cpc"] = cpc_sum / len(keywords) if keywords else 0
        
        return {
            "keywords": keywords,
            "metrics": metrics
        }
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to integer"""
        try:
            if pd.isna(value) or value == "":
                return 0
            
            if isinstance(value, str):
                # Remove commas and other formatting
                value = value.replace(",", "").replace("$", "").replace("%", "")
            
            return max(0, int(float(value)))
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == "":
                return 0.0
            
            if isinstance(value, str):
                # Remove formatting
                value = value.replace(",", "").replace("$", "").replace("%", "")
            
            return max(0.0, float(value))
        except (ValueError, TypeError):
            return 0.0
    
    def _determine_intent(self, keyword: str) -> str:
        """Determine search intent from keyword"""
        keyword_lower = keyword.lower()
        
        informational_keywords = ["how", "what", "why", "when", "where", "guide", "tutorial", "learn"]
        commercial_keywords = ["buy", "purchase", "price", "cost", "best", "review", "compare", "top"]
        transactional_keywords = ["buy", "purchase", "order", "shop", "deal", "offer", "discount", "sale"]
        
        if any(word in keyword_lower for word in transactional_keywords):
            return "transactional"
        elif any(word in keyword_lower for word in commercial_keywords):
            return "commercial"
        elif any(word in keyword_lower for word in informational_keywords):
            return "informational"
        else:
            return "informational"  # Default
    
    def _calculate_priority_score(self, keyword_data: Dict[str, Any]) -> float:
        """Calculate priority score for a keyword"""
        search_volume = keyword_data.get("search_volume", 0)
        difficulty = keyword_data.get("difficulty", 50)
        cpc = keyword_data.get("cpc", 0)
        
        # Normalize values
        volume_score = min(search_volume / 10000, 1.0)
        difficulty_score = 1.0 - (difficulty / 100)
        cpc_score = min(cpc / 10, 1.0)
        
        # Calculate weighted score
        priority_score = (volume_score * 0.4 + difficulty_score * 0.3 + cpc_score * 0.3)
        
        return min(max(priority_score, 0.0), 1.0)
    
    async def detect_tool_format(self, file_content: bytes) -> Dict[str, Any]:
        """Detect external tool format from CSV file"""
        try:
            df = pd.read_csv(io.BytesIO(file_content), nrows=5)
            
            # Tool-specific column patterns
            tool_patterns = {
                "ahrefs": ["Keyword", "Search Volume", "KD", "CPC", "Competition"],
                "semrush": ["Keyword", "Search Volume", "KD", "CPC", "Competition"],
                "ubersuggest": ["Keyword", "Search Volume", "SEO Difficulty", "CPC", "Competition"]
            }
            
            detected_tools = {}
            
            for tool, patterns in tool_patterns.items():
                score = 0
                for pattern in patterns:
                    if any(pattern.lower() in col.lower() for col in df.columns):
                        score += 1
                
                detected_tools[tool] = score / len(patterns)
            
            best_tool = max(detected_tools, key=detected_tools.get) if detected_tools else None
            confidence = detected_tools[best_tool] if best_tool else 0
            
            return {
                "detected_tool": best_tool,
                "confidence": confidence,
                "available_columns": list(df.columns),
                "sample_data": df.head(3).to_dict("records")
            }
            
        except Exception as e:
            logger.error("Failed to detect tool format", error=str(e))
            return {
                "detected_tool": None,
                "confidence": 0,
                "available_columns": [],
                "sample_data": []
            }
    
    async def export_keywords_for_tool(
        self, 
        keywords: List[Dict[str, Any]], 
        tool_type: str,
        include_metrics: bool = True
    ) -> str:
        """Export keywords in format suitable for external tools"""
        try:
            if tool_type == "ahrefs":
                return await self._export_for_ahrefs(keywords, include_metrics)
            elif tool_type == "semrush":
                return await self._export_for_semrush(keywords, include_metrics)
            elif tool_type == "ubersuggest":
                return await self._export_for_ubersuggest(keywords, include_metrics)
            else:
                return await self._export_generic_csv(keywords, include_metrics)
                
        except Exception as e:
            logger.error("Failed to export keywords", error=str(e))
            raise
    
    async def _export_for_ahrefs(self, keywords: List[Dict[str, Any]], include_metrics: bool) -> str:
        """Export keywords in Ahrefs format"""
        df_data = []
        
        for kw in keywords:
            row = {
                "Keyword": kw.get("keyword", ""),
                "Search Volume": kw.get("search_volume", 0),
                "KD": kw.get("difficulty", 0),
                "CPC": kw.get("cpc", 0),
                "Competition": kw.get("competition", "medium")
            }
            
            if include_metrics:
                row.update({
                    "Intent": kw.get("intent", "informational"),
                    "Priority Score": kw.get("priority_score", 0)
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)
    
    async def _export_for_semrush(self, keywords: List[Dict[str, Any]], include_metrics: bool) -> str:
        """Export keywords in Semrush format"""
        df_data = []
        
        for kw in keywords:
            row = {
                "Keyword": kw.get("keyword", ""),
                "Search Volume": kw.get("search_volume", 0),
                "KD": kw.get("difficulty", 0),
                "CPC": kw.get("cpc", 0),
                "Competition": kw.get("competition", "medium")
            }
            
            if include_metrics:
                row.update({
                    "Intent": kw.get("intent", "informational"),
                    "Priority Score": kw.get("priority_score", 0)
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)
    
    async def _export_for_ubersuggest(self, keywords: List[Dict[str, Any]], include_metrics: bool) -> str:
        """Export keywords in Ubersuggest format"""
        df_data = []
        
        for kw in keywords:
            row = {
                "Keyword": kw.get("keyword", ""),
                "Search Volume": kw.get("search_volume", 0),
                "SEO Difficulty": kw.get("difficulty", 0),
                "CPC": kw.get("cpc", 0),
                "Competition": kw.get("competition", "medium")
            }
            
            if include_metrics:
                row.update({
                    "Intent": kw.get("intent", "informational"),
                    "Priority Score": kw.get("priority_score", 0)
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)
    
    async def _export_generic_csv(self, keywords: List[Dict[str, Any]], include_metrics: bool) -> str:
        """Export keywords in generic CSV format"""
        df_data = []
        
        for kw in keywords:
            row = {
                "keyword": kw.get("keyword", ""),
                "search_volume": kw.get("search_volume", 0),
                "difficulty": kw.get("difficulty", 0),
                "cpc": kw.get("cpc", 0),
                "competition": kw.get("competition", "medium")
            }
            
            if include_metrics:
                row.update({
                    "intent": kw.get("intent", "informational"),
                    "priority_score": kw.get("priority_score", 0)
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)
