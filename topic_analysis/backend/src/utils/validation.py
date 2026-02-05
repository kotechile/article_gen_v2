"""
Validation Utility

Validates file uploads, data formats, and API inputs.
Handles TSV parsing, data validation, and error reporting.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationUtility:
    """Utility for validating various inputs and data formats"""
    
    # File size limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_KEYWORDS = 50000
    
    # Required TSV columns for Ahrefs export
    REQUIRED_COLUMNS = ['Keyword', 'Volume', 'Difficulty', 'CPC', 'Intents']
    
    def __init__(self):
        self.supported_file_types = ['.tsv', '.csv']
    
    def validate_file_upload(
        self, 
        file_path: str, 
        file_size: int,
        file_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate file upload requirements
        
        Args:
            file_path: Path to the uploaded file
            file_size: Size of the file in bytes
            file_type: MIME type of the file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check file size
            if file_size > self.MAX_FILE_SIZE:
                errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE} bytes)")
            
            # Check file extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_file_types:
                errors.append(f"Unsupported file type: {file_extension}. Supported types: {', '.join(self.supported_file_types)}")
            
            # Check if file exists
            if not Path(file_path).exists():
                errors.append(f"File not found: {file_path}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating file upload: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def validate_tsv_format(self, file_path: str) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
        """
        Validate TSV file format and structure
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            Tuple of (is_valid, error_messages, dataframe)
        """
        errors = []
        dataframe = None
        
        try:
            # Try to read the TSV file
            dataframe = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            
            # Check if file is empty
            if dataframe.empty:
                errors.append("File is empty")
                return False, errors, None
            
            # Check for required columns
            missing_columns = set(self.REQUIRED_COLUMNS) - set(dataframe.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Check for minimum number of rows
            if len(dataframe) < 1:
                errors.append("File must contain at least 1 row of data")
            
            # Check for maximum number of rows
            if len(dataframe) > self.MAX_KEYWORDS:
                errors.append(f"File contains too many keywords ({len(dataframe)}). Maximum allowed: {self.MAX_KEYWORDS}")
            
            # Validate data types and ranges
            validation_errors = self._validate_dataframe_content(dataframe)
            errors.extend(validation_errors)
            
            return len(errors) == 0, errors, dataframe
            
        except pd.errors.EmptyDataError:
            errors.append("File is empty or contains no data")
            return False, errors, None
        except pd.errors.ParserError as e:
            errors.append(f"Error parsing TSV file: {str(e)}")
            return False, errors, None
        except Exception as e:
            logger.error(f"Error validating TSV format: {str(e)}")
            errors.append(f"Unexpected error: {str(e)}")
            return False, errors, None
    
    def _validate_dataframe_content(self, df: pd.DataFrame) -> List[str]:
        """Validate the content of the dataframe"""
        errors = []
        
        try:
            # Validate Keyword column
            if 'Keyword' in df.columns:
                if df['Keyword'].isna().any():
                    errors.append("Keyword column contains empty values")
                
                # Check for duplicate keywords
                duplicates = df['Keyword'].duplicated().sum()
                if duplicates > 0:
                    errors.append(f"Found {duplicates} duplicate keywords")
            
            # Validate Volume column
            if 'Volume' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Volume']):
                    errors.append("Volume column must contain numeric values")
                else:
                    # Check for negative volumes
                    negative_volumes = (df['Volume'] < 0).sum()
                    if negative_volumes > 0:
                        errors.append(f"Found {negative_volumes} negative volume values")
            
            # Validate Difficulty column
            if 'Difficulty' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Difficulty']):
                    errors.append("Difficulty column must contain numeric values")
                else:
                    # Check for difficulty values outside 0-100 range
                    invalid_difficulty = ((df['Difficulty'] < 0) | (df['Difficulty'] > 100)).sum()
                    if invalid_difficulty > 0:
                        errors.append(f"Found {invalid_difficulty} difficulty values outside 0-100 range")
            
            # Validate CPC column
            if 'CPC' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['CPC']):
                    errors.append("CPC column must contain numeric values")
                else:
                    # Check for negative CPC values
                    negative_cpc = (df['CPC'] < 0).sum()
                    if negative_cpc > 0:
                        errors.append(f"Found {negative_cpc} negative CPC values")
            
            # Validate Intents column
            if 'Intents' in df.columns:
                # Check for valid intent values
                valid_intents = {'Informational', 'Commercial', 'Navigational', 'Transactional'}
                invalid_intents = 0
                
                for intents_str in df['Intents'].dropna():
                    if isinstance(intents_str, str):
                        intents = [intent.strip() for intent in intents_str.split(',')]
                        for intent in intents:
                            if intent and intent not in valid_intents:
                                invalid_intents += 1
                                break
                
                if invalid_intents > 0:
                    errors.append(f"Found {invalid_intents} rows with invalid intent values")
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating dataframe content: {str(e)}")
            return [f"Content validation error: {str(e)}"]
    
    def validate_keyword_data(self, keyword_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate individual keyword data
        
        Args:
            keyword_data: Dictionary containing keyword data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = ['keyword', 'volume', 'difficulty', 'cpc']
            for field in required_fields:
                if field not in keyword_data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate keyword
            if 'keyword' in keyword_data:
                keyword = keyword_data['keyword']
                if not isinstance(keyword, str) or not keyword.strip():
                    errors.append("Keyword must be a non-empty string")
                elif len(keyword) > 255:
                    errors.append("Keyword must be 255 characters or less")
            
            # Validate volume
            if 'volume' in keyword_data:
                volume = keyword_data['volume']
                if not isinstance(volume, (int, float)) or volume < 0:
                    errors.append("Volume must be a non-negative number")
            
            # Validate difficulty
            if 'difficulty' in keyword_data:
                difficulty = keyword_data['difficulty']
                if not isinstance(difficulty, (int, float)) or not (0 <= difficulty <= 100):
                    errors.append("Difficulty must be a number between 0 and 100")
            
            # Validate CPC
            if 'cpc' in keyword_data:
                cpc = keyword_data['cpc']
                if not isinstance(cpc, (int, float)) or cpc < 0:
                    errors.append("CPC must be a non-negative number")
            
            # Validate intents
            if 'intents' in keyword_data:
                intents = keyword_data['intents']
                if not isinstance(intents, list):
                    errors.append("Intents must be a list")
                else:
                    valid_intents = {'Informational', 'Commercial', 'Navigational', 'Transactional'}
                    for intent in intents:
                        if not isinstance(intent, str) or intent not in valid_intents:
                            errors.append(f"Invalid intent: {intent}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating keyword data: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def validate_api_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate API input data
        
        Args:
            input_data: Dictionary containing API input
            required_fields: List of required field names
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check for required fields
            for field in required_fields:
                if field not in input_data:
                    errors.append(f"Missing required field: {field}")
                elif input_data[field] is None:
                    errors.append(f"Field cannot be null: {field}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating API input: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def sanitize_keyword(self, keyword: str) -> str:
        """
        Sanitize keyword string
        
        Args:
            keyword: Raw keyword string
            
        Returns:
            Sanitized keyword string
        """
        try:
            if not isinstance(keyword, str):
                return ""
            
            # Remove extra whitespace
            sanitized = keyword.strip()
            
            # Remove special characters that might cause issues
            sanitized = re.sub(r'[^\w\s-]', '', sanitized)
            
            # Limit length
            sanitized = sanitized[:255]
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing keyword: {str(e)}")
            return ""
    
    def validate_file_retention(self, file_created_at: datetime) -> bool:
        """
        Validate if file is within retention period (90 days)
        
        Args:
            file_created_at: When the file was created
            
        Returns:
            True if file is within retention period
        """
        try:
            from datetime import timedelta
            
            retention_period = timedelta(days=90)
            cutoff_date = datetime.utcnow() - retention_period
            
            return file_created_at >= cutoff_date
            
        except Exception as e:
            logger.error(f"Error validating file retention: {str(e)}")
            return False
