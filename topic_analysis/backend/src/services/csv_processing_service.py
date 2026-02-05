"""
CSV Processing Service for enhanced research workflow
"""

import pandas as pd
import io
import json
import uuid
from typing import Dict, List, Any, Optional, BinaryIO
from datetime import datetime
import structlog
from ..core.config import get_settings
from ..models.trend_analysis import TrendAnalysis, CSVProcessingStatus

logger = structlog.get_logger()
settings = get_settings()

class CSVProcessingService:
    """Service for processing CSV files with trend data"""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size
        self.csv_max_rows = settings.csv_max_rows
        self.csv_processing_timeout = settings.csv_processing_timeout
        self.allowed_file_types = settings.allowed_file_types
    
    async def process_trend_csv(
        self, 
        file_content: bytes, 
        column_mapping: Dict[str, str],
        user_id: str,
        trend_analysis_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process CSV file with trend data"""
        try:
            logger.info("Processing trend CSV file", user_id=user_id, file_size=len(file_content))
            
            # Validate file size
            if len(file_content) > self.max_file_size:
                raise ValueError(f"File size exceeds maximum limit of {self.max_file_size} bytes")
            
            # Read CSV file
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Validate row count
            if len(df) > self.csv_max_rows:
                raise ValueError(f"CSV has too many rows. Maximum allowed: {self.csv_max_rows}")
            
            # Validate required columns
            required_columns = ["trend_name"]
            missing_columns = [col for col in required_columns if column_mapping.get(col) not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Process and validate data
            processed_data = await self._process_trend_data(df, column_mapping)
            
            # Create trend analysis record if not provided
            if not trend_analysis_id:
                trend_analysis = TrendAnalysis(
                    user_id=user_id,
                    topics=processed_data["trends"],
                    csv_upload_id=str(uuid.uuid4()),
                    csv_columns=column_mapping,
                    csv_row_count=len(df),
                    csv_processing_status=CSVProcessingStatus.COMPLETED
                )
            else:
                # Update existing trend analysis
                trend_analysis = self.db.get_TrendAnalysis_by_id(
                    TrendAnalysis.id == trend_analysis_id,
                    TrendAnalysis.user_id == user_id
                )
                
                if trend_analysis:
                    trend_analysis.csv_upload_id = str(uuid.uuid4())
                    trend_analysis.csv_columns = column_mapping
                    trend_analysis.csv_row_count = len(df)
                    trend_analysis.csv_processing_status = CSVProcessingStatus.COMPLETED
                    trend_analysis.topics = processed_data["trends"]
            
            logger.info("CSV processing completed successfully", 
                       user_id=user_id, 
                       trends_count=len(processed_data["trends"]))
            
            return {
                "upload_id": trend_analysis.csv_upload_id,
                "processing_status": "completed",
                "row_count": len(df),
                "trends_available": len(processed_data["trends"]),
                "message": "CSV processed successfully",
                "trends": processed_data["trends"],
                "validation_warnings": processed_data["warnings"]
            }
            
        except Exception as e:
            logger.error("Failed to process CSV file", error=str(e), user_id=user_id)
            raise
    
    async def _process_trend_data(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process and validate trend data from DataFrame"""
        trends = []
        warnings = []
        
        # Map columns
        trend_name_col = column_mapping.get("trend_name")
        trend_description_col = column_mapping.get("trend_description")
        trend_category_col = column_mapping.get("trend_category")
        search_volume_col = column_mapping.get("search_volume")
        competition_level_col = column_mapping.get("competition_level")
        date_col = column_mapping.get("date")
        
        for index, row in df.iterrows():
            try:
                # Extract trend data
                trend_data = {
                    "id": f"trend_{index}",
                    "trend_name": str(row[trend_name_col]).strip() if trend_name_col in row else f"Trend {index + 1}",
                    "trend_description": str(row[trend_description_col]).strip() if trend_description_col in row else "",
                    "trend_category": self._normalize_category(
                        str(row[trend_category_col]).strip() if trend_category_col in row else "technology"
                    ),
                    "search_volume": self._normalize_search_volume(
                        row[search_volume_col] if search_volume_col in row else 0
                    ),
                    "competition_level": self._normalize_competition_level(
                        str(row[competition_level_col]).strip() if competition_level_col in row else "medium"
                    ),
                    "source": "csv_upload",
                    "date": self._normalize_date(
                        row[date_col] if date_col in row else datetime.now().isoformat()
                    )
                }
                
                # Validate trend data
                validation_warnings = self._validate_trend_data(trend_data, index)
                warnings.extend(validation_warnings)
                
                trends.append(trend_data)
                
            except Exception as e:
                warnings.append(f"Row {index + 1}: Error processing trend data - {str(e)}")
                continue
        
        return {
            "trends": trends,
            "warnings": warnings
        }
    
    def _normalize_category(self, category: str) -> str:
        """Normalize trend category"""
        category_lower = category.lower()
        valid_categories = ["technology", "business", "lifestyle", "health", "finance", "entertainment"]
        
        for valid_cat in valid_categories:
            if valid_cat in category_lower:
                return valid_cat
        
        return "technology"  # Default category
    
    def _normalize_search_volume(self, volume: Any) -> int:
        """Normalize search volume to integer"""
        try:
            if pd.isna(volume) or volume == "":
                return 0
            
            # Handle string numbers with commas
            if isinstance(volume, str):
                volume = volume.replace(",", "")
            
            return max(0, int(float(volume)))
        except (ValueError, TypeError):
            return 0
    
    def _normalize_competition_level(self, level: str) -> str:
        """Normalize competition level"""
        level_lower = level.lower()
        
        if any(word in level_lower for word in ["low", "easy", "simple"]):
            return "low"
        elif any(word in level_lower for word in ["high", "hard", "difficult", "competitive"]):
            return "high"
        else:
            return "medium"
    
    def _normalize_date(self, date: Any) -> str:
        """Normalize date to ISO format"""
        try:
            if pd.isna(date) or date == "":
                return datetime.now().isoformat()
            
            # Try to parse various date formats
            if isinstance(date, str):
                # Common date formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            # If it's already a datetime object
            if isinstance(date, datetime):
                return date.isoformat()
            
            # Try pandas to_datetime
            parsed_date = pd.to_datetime(date)
            return parsed_date.isoformat()
            
        except Exception:
            return datetime.now().isoformat()
    
    def _validate_trend_data(self, trend_data: Dict[str, Any], row_index: int) -> List[str]:
        """Validate individual trend data and return warnings"""
        warnings = []
        
        # Validate trend name
        if not trend_data["trend_name"] or len(trend_data["trend_name"]) < 3:
            warnings.append(f"Row {row_index + 1}: Trend name is too short or empty")
        
        if len(trend_data["trend_name"]) > 255:
            warnings.append(f"Row {row_index + 1}: Trend name is too long (max 255 characters)")
        
        # Validate search volume
        if trend_data["search_volume"] < 0:
            warnings.append(f"Row {row_index + 1}: Search volume cannot be negative")
        
        # Validate trend description length
        if trend_data["trend_description"] and len(trend_data["trend_description"]) > 1000:
            warnings.append(f"Row {row_index + 1}: Trend description is too long (max 1000 characters)")
        
        return warnings
    
    async def detect_csv_format(self, file_content: bytes) -> Dict[str, Any]:
        """Detect CSV format and suggest column mapping"""
        try:
            # Read first few rows to analyze structure
            df = pd.read_csv(io.BytesIO(file_content), nrows=5)
            
            # Common column name patterns
            column_patterns = {
                "trend_name": ["trend", "name", "title", "topic", "keyword"],
                "trend_description": ["description", "desc", "summary", "details"],
                "trend_category": ["category", "cat", "type", "niche"],
                "search_volume": ["volume", "searches", "search_volume", "traffic"],
                "competition_level": ["competition", "comp", "difficulty", "level"],
                "date": ["date", "created", "timestamp", "time"]
            }
            
            suggested_mapping = {}
            confidence_scores = {}
            
            for field, patterns in column_patterns.items():
                best_match = None
                best_score = 0
                
                for col in df.columns:
                    col_lower = col.lower()
                    for pattern in patterns:
                        if pattern in col_lower:
                            score = len(pattern) / len(col_lower)
                            if score > best_score:
                                best_score = score
                                best_match = col
                
                if best_match and best_score > 0.3:
                    suggested_mapping[field] = best_match
                    confidence_scores[field] = best_score
            
            return {
                "suggested_mapping": suggested_mapping,
                "confidence_scores": confidence_scores,
                "available_columns": list(df.columns),
                "sample_data": df.head(3).to_dict("records")
            }
            
        except Exception as e:
            logger.error("Failed to detect CSV format", error=str(e))
            raise
    
    async def validate_csv_structure(self, file_content: bytes, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Validate CSV structure before processing"""
        try:
            df = pd.read_csv(io.BytesIO(file_content), nrows=10)  # Read first 10 rows for validation
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "row_count": len(df),
                "column_count": len(df.columns)
            }
            
            # Check required columns
            required_columns = ["trend_name"]
            for req_col in required_columns:
                if req_col not in column_mapping or column_mapping[req_col] not in df.columns:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Required column '{req_col}' not found")
            
            # Check for empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                validation_result["warnings"].append(f"Found {empty_rows} empty rows")
            
            # Check data types
            for field, col_name in column_mapping.items():
                if col_name in df.columns:
                    if field == "search_volume":
                        try:
                            pd.to_numeric(df[col_name], errors='raise')
                        except (ValueError, TypeError):
                            validation_result["warnings"].append(f"Column '{col_name}' contains non-numeric values for search volume")
            
            return validation_result
            
        except Exception as e:
            logger.error("Failed to validate CSV structure", error=str(e))
            return {
                "is_valid": False,
                "errors": [f"CSV parsing error: {str(e)}"],
                "warnings": [],
                "row_count": 0,
                "column_count": 0
            }
