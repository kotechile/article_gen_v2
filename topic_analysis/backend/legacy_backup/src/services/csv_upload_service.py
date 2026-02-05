"""
CSV Upload Service for Trend Data
Handles CSV file uploads and processing for trend analysis
"""

import csv
import io
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
from ..core.redis import cache

logger = structlog.get_logger()

class CSVUploadService:
    """Service for handling CSV uploads and trend data processing"""
    
    def __init__(self, db: Session):
        self.db = db
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_file_size = 10 * 1024 * 1024  # 10MB max file size
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        
        # Expected column mappings for different tools
        self.column_mappings = {
            'google_trends': {
                'keyword': ['keyword', 'term', 'query', 'search_term'],
                'search_volume': ['search_volume', 'volume', 'searches', 'monthly_searches'],
                'trend_score': ['trend_score', 'score', 'interest', 'trend_value'],
                'growth_rate': ['growth_rate', 'growth', 'change', 'trend_change'],
                'date': ['date', 'time', 'timestamp', 'period']
            },
            'semrush': {
                'keyword': ['keyword', 'term', 'query', 'search_term'],
                'search_volume': ['search_volume', 'volume', 'searches', 'monthly_searches'],
                'cpc': ['cpc', 'cost_per_click', 'price'],
                'competition': ['competition', 'competition_level', 'difficulty'],
                'trend': ['trend', 'trend_score', 'interest']
            },
            'ahrefs': {
                'keyword': ['keyword', 'term', 'query', 'search_term'],
                'search_volume': ['search_volume', 'volume', 'searches', 'monthly_searches'],
                'kd': ['kd', 'keyword_difficulty', 'difficulty'],
                'cpc': ['cpc', 'cost_per_click', 'price'],
                'trend': ['trend', 'trend_score', 'interest']
            },
            'ubersuggest': {
                'keyword': ['keyword', 'term', 'query', 'search_term'],
                'search_volume': ['search_volume', 'volume', 'searches', 'monthly_searches'],
                'cpc': ['cpc', 'cost_per_click', 'price'],
                'competition': ['competition', 'competition_level', 'difficulty'],
                'trend': ['trend', 'trend_score', 'interest']
            }
        }
    
    async def process_csv_upload(
        self,
        file: UploadFile,
        user_id: str,
        tool_type: str = 'google_trends',
        workflow_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded CSV file and extract trend data
        
        Args:
            file: Uploaded file
            user_id: User ID
            tool_type: Type of tool (google_trends, semrush, ahrefs, ubersuggest)
            workflow_session_id: Optional workflow session ID
            
        Returns:
            Dict containing processed trend data
        """
        try:
            logger.info("Processing CSV upload", 
                       filename=file.filename, 
                       user_id=user_id,
                       tool_type=tool_type)
            
            # Validate file
            await self._validate_file(file)
            
            # Read file content
            content = await file.read()
            
            # Parse CSV based on file extension
            if file.filename.endswith('.csv'):
                df = await self._parse_csv(content)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = await self._parse_excel(content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # Validate and clean data
            df_cleaned = await self._clean_dataframe(df, tool_type)
            
            # Extract trend data
            trend_data = await self._extract_trend_data(df_cleaned, tool_type)
            
            # Store in cache
            cache_key = f"csv_upload:{user_id}:{workflow_session_id or 'default'}"
            await self._cache_trend_data(cache_key, trend_data)
            
            logger.info("CSV upload processed successfully", 
                       filename=file.filename,
                       rows_processed=len(df_cleaned),
                       trends_extracted=len(trend_data.get('trends', [])))
            
            return {
                "success": True,
                "filename": file.filename,
                "rows_processed": len(df_cleaned),
                "trends_extracted": len(trend_data.get('trends', [])),
                "tool_type": tool_type,
                "trend_data": trend_data,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to process CSV upload", 
                        filename=file.filename,
                        user_id=user_id,
                        error=str(e))
            raise HTTPException(status_code=500, detail="Failed to process CSV file")
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in self.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Check file size
        if hasattr(file, 'size') and file.size > self.max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
            )
    
    async def _parse_csv(self, content: bytes) -> pd.DataFrame:
        """Parse CSV content into DataFrame"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    text_content = content.decode(encoding)
                    df = pd.read_csv(io.StringIO(text_content))
                    if not df.empty:
                        logger.info("CSV parsed successfully", encoding=encoding, rows=len(df))
                        return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with errors='ignore'
            text_content = content.decode('utf-8', errors='ignore')
            df = pd.read_csv(io.StringIO(text_content))
            logger.warning("CSV parsed with encoding errors", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to parse CSV", error=str(e))
            raise HTTPException(status_code=400, detail="Invalid CSV file format")
    
    async def _parse_excel(self, content: bytes) -> pd.DataFrame:
        """Parse Excel content into DataFrame"""
        try:
            df = pd.read_excel(io.BytesIO(content))
            if df.empty:
                raise HTTPException(status_code=400, detail="Empty Excel file")
            
            logger.info("Excel file parsed successfully", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to parse Excel file", error=str(e))
            raise HTTPException(status_code=400, detail="Invalid Excel file format")
    
    async def _clean_dataframe(self, df: pd.DataFrame, tool_type: str) -> pd.DataFrame:
        """Clean and validate DataFrame"""
        try:
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Remove rows where all values are NaN
            df = df.dropna(how='all')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Validate required columns exist
            required_columns = self._get_required_columns(tool_type)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning("Missing required columns", 
                             missing=missing_columns,
                             available=list(df.columns))
                # Try to map similar column names
                df = await self._map_column_names(df, tool_type)
            
            logger.info("DataFrame cleaned successfully", 
                       rows=len(df), 
                       columns=list(df.columns))
            
            return df
            
        except Exception as e:
            logger.error("Failed to clean DataFrame", error=str(e))
            raise HTTPException(status_code=400, detail="Failed to clean data")
    
    async def _map_column_names(self, df: pd.DataFrame, tool_type: str) -> pd.DataFrame:
        """Map column names to expected format"""
        try:
            column_mapping = self.column_mappings.get(tool_type, {})
            mapped_columns = {}
            
            for expected_col, possible_names in column_mapping.items():
                for col in df.columns:
                    if any(name in col.lower() for name in possible_names):
                        mapped_columns[col] = expected_col
                        break
            
            # Rename columns
            df = df.rename(columns=mapped_columns)
            
            logger.info("Column names mapped", mapping=mapped_columns)
            return df
            
        except Exception as e:
            logger.error("Failed to map column names", error=str(e))
            return df
    
    def _get_required_columns(self, tool_type: str) -> List[str]:
        """Get required columns for tool type"""
        base_columns = ['keyword']
        
        if tool_type == 'google_trends':
            return base_columns + ['search_volume', 'trend_score']
        elif tool_type in ['semrush', 'ahrefs', 'ubersuggest']:
            return base_columns + ['search_volume', 'cpc', 'competition']
        else:
            return base_columns + ['search_volume']
    
    async def _extract_trend_data(self, df: pd.DataFrame, tool_type: str) -> Dict[str, Any]:
        """Extract trend data from cleaned DataFrame"""
        try:
            trends = []
            
            for _, row in df.iterrows():
                trend = {
                    "keyword": str(row.get('keyword', '')),
                    "search_volume": self._safe_int(row.get('search_volume', 0)),
                    "source": tool_type,
                    "raw_data": row.to_dict()
                }
                
                # Add tool-specific data
                if tool_type == 'google_trends':
                    trend.update({
                        "trend_score": self._safe_int(row.get('trend_score', 50)),
                        "growth_rate": self._safe_float(row.get('growth_rate', 0.0)),
                        "seasonality": self._analyze_seasonality(row),
                        "related_queries": self._extract_related_queries(row)
                    })
                elif tool_type in ['semrush', 'ahrefs', 'ubersuggest']:
                    trend.update({
                        "cpc": self._safe_float(row.get('cpc', 0.0)),
                        "competition": self._safe_float(row.get('competition', 0.0)),
                        "difficulty": self._safe_float(row.get('difficulty', 0.0)),
                        "trend": self._safe_float(row.get('trend', 0.0))
                    })
                
                # Add time series if date column exists
                if 'date' in row and pd.notna(row['date']):
                    trend["date"] = self._format_date(row['date'])
                
                trends.append(trend)
            
            return {
                "trends": trends,
                "source": tool_type,
                "total_keywords": len(trends),
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to extract trend data", error=str(e))
            raise HTTPException(status_code=400, detail="Failed to extract trend data")
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int"""
        try:
            if pd.isna(value) or value == '':
                return 0
            return int(float(str(value)))
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == '':
                return 0.0
            return float(str(value))
        except (ValueError, TypeError):
            return 0.0
    
    def _analyze_seasonality(self, row: pd.Series) -> str:
        """Analyze seasonality from row data"""
        try:
            # Simple seasonality analysis based on available data
            trend_score = self._safe_int(row.get('trend_score', 50))
            growth_rate = self._safe_float(row.get('growth_rate', 0.0))
            
            if abs(growth_rate) > 20:
                return "high"
            elif abs(growth_rate) > 10:
                return "moderate"
            else:
                return "stable"
        except:
            return "stable"
    
    def _extract_related_queries(self, row: pd.Series) -> List[str]:
        """Extract related queries from row data"""
        try:
            related_queries = []
            
            # Look for related query columns
            for col in row.index:
                if 'related' in col.lower() or 'query' in col.lower():
                    value = row[col]
                    if pd.notna(value) and value != '':
                        if isinstance(value, str):
                            # Split by common delimiters
                            queries = [q.strip() for q in value.split(',') if q.strip()]
                            related_queries.extend(queries[:5])  # Limit to 5
            
            return related_queries[:5]  # Return max 5 queries
        except:
            return []
    
    def _format_date(self, date_value: Any) -> str:
        """Format date value to ISO string"""
        try:
            if pd.isna(date_value):
                return datetime.utcnow().isoformat()
            
            if isinstance(date_value, str):
                # Try to parse common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        dt = datetime.strptime(date_value, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
            
            # If it's already a datetime object
            if hasattr(date_value, 'isoformat'):
                return date_value.isoformat()
            
            return datetime.utcnow().isoformat()
        except:
            return datetime.utcnow().isoformat()
    
    async def _cache_trend_data(self, cache_key: str, trend_data: Dict[str, Any]) -> None:
        """Cache trend data in Redis"""
        try:
            cache.setex(cache_key, self.cache_ttl, json.dumps(trend_data))
            logger.info("Trend data cached successfully", cache_key=cache_key)
        except Exception as e:
            logger.warning("Failed to cache trend data", cache_key=cache_key, error=str(e))
    
    async def get_cached_trend_data(self, user_id: str, workflow_session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached trend data"""
        try:
            cache_key = f"csv_upload:{user_id}:{workflow_session_id or 'default'}"
            cached_data = cache.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning("Failed to get cached trend data", cache_key=cache_key, error=str(e))
            return None
    
    async def validate_csv_structure(self, file: UploadFile, tool_type: str) -> Dict[str, Any]:
        """Validate CSV structure without processing full file"""
        try:
            # Read first few rows to validate structure
            content = await file.read()
            file.file.seek(0)  # Reset file pointer
            
            if file.filename.endswith('.csv'):
                df = await self._parse_csv(content)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = await self._parse_excel(content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Check required columns
            required_columns = self._get_required_columns(tool_type)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            # Try to map similar column names
            mapped_columns = {}
            if missing_columns:
                column_mapping = self.column_mappings.get(tool_type, {})
                for expected_col, possible_names in column_mapping.items():
                    for col in df.columns:
                        if any(name in col.lower() for name in possible_names):
                            mapped_columns[col] = expected_col
                            break
            
            return {
                "valid": len(missing_columns) == 0 or len(mapped_columns) > 0,
                "columns": list(df.columns),
                "required_columns": required_columns,
                "missing_columns": missing_columns,
                "mapped_columns": mapped_columns,
                "sample_rows": df.head(3).to_dict('records') if not df.empty else []
            }
            
        except Exception as e:
            logger.error("Failed to validate CSV structure", error=str(e))
            raise HTTPException(status_code=400, detail="Failed to validate CSV structure")
