"""
File parser service for processing Ahrefs TSV files
"""

import pandas as pd
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from ..models.keyword import Keyword, UploadedFile
from ..config import settings

logger = logging.getLogger(__name__)

class FileParserService:
    """Service for parsing Ahrefs TSV files"""
    
    def __init__(self):
        self.required_columns = [
            'Keyword', 'Volume', 'Difficulty', 'CPC', 'Intents'
        ]
        self.column_mapping = {
            'Keyword': 'keyword',
            'Volume': 'search_volume',
            'Difficulty': 'keyword_difficulty',
            'CPC': 'cpc',
            'Intents': 'intents'
        }
    
    def parse_tsv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse TSV file and return list of keyword dictionaries
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            List of keyword dictionaries
            
        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Read TSV file
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                encoding='utf-8',
                na_values=['', 'N/A', 'n/a', 'NULL', 'null']
            )
            
            # Validate required columns
            self._validate_columns(df)
            
            # Clean and process data
            df = self._clean_dataframe(df)
            
            # Convert to list of dictionaries
            keywords = df.to_dict('records')
            
            logger.info(f"Successfully parsed {len(keywords)} keywords from {file_path}")
            return keywords
            
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or contains no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"Invalid TSV format: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise ValueError(f"Error parsing file: {str(e)}")
    
    def parse_tsv_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse TSV content from string
        
        Args:
            content: TSV content as string
            
        Returns:
            List of keyword dictionaries
        """
        try:
            # Read from string
            df = pd.read_csv(
                io.StringIO(content),
                sep='\t',
                encoding='utf-8',
                na_values=['', 'N/A', 'n/a', 'NULL', 'null']
            )
            
            # Validate required columns
            self._validate_columns(df)
            
            # Clean and process data
            df = self._clean_dataframe(df)
            
            # Convert to list of dictionaries
            keywords = df.to_dict('records')
            
            logger.info(f"Successfully parsed {len(keywords)} keywords from content")
            return keywords
            
        except Exception as e:
            logger.error(f"Error parsing TSV content: {str(e)}")
            raise ValueError(f"Error parsing TSV content: {str(e)}")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist"""
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process dataframe"""
        # Remove rows with missing required data
        df = df.dropna(subset=['Keyword'])
        
        # Fill missing values with defaults
        df['Volume'] = df['Volume'].fillna(0)
        df['Difficulty'] = df['Difficulty'].fillna(0)
        df['CPC'] = df['CPC'].fillna(0.0)
        df['Intents'] = df['Intents'].fillna('')
        
        # Convert data types
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        df['Difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce').fillna(0).astype(float)
        df['CPC'] = pd.to_numeric(df['CPC'], errors='coerce').fillna(0.0).astype(float)
        
        # Clean keyword strings
        df['Keyword'] = df['Keyword'].astype(str).str.strip()
        
        # Clean intent strings
        df['Intents'] = df['Intents'].astype(str).str.strip()
        
        # Remove empty keywords
        df = df[df['Keyword'].str.len() > 0]
        
        # Limit to maximum keywords
        if len(df) > settings.max_keywords_per_file:
            df = df.head(settings.max_keywords_per_file)
            logger.warning(f"Limited to {settings.max_keywords_per_file} keywords")
        
        return df
    
    def validate_file_size(self, file_path: str) -> bool:
        """Validate file size is within limits"""
        file_size = Path(file_path).stat().st_size
        return file_size <= settings.max_file_size
    
    def validate_file_type(self, file_path: str) -> bool:
        """Validate file type is TSV"""
        return file_path.lower().endswith(('.tsv', '.txt'))
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        path = Path(file_path)
        return {
            'filename': path.name,
            'file_size': path.stat().st_size,
            'file_extension': path.suffix,
            'is_valid_type': self.validate_file_type(file_path),
            'is_valid_size': self.validate_file_size(file_path)
        }
    
    def extract_keyword_insights(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract insights from keyword data"""
        if not keywords:
            return {}
        
        df = pd.DataFrame(keywords)
        
        insights = {
            'total_keywords': len(keywords),
            'total_search_volume': int(df['Volume'].sum()),
            'average_difficulty': float(df['Difficulty'].mean()),
            'average_cpc': float(df['CPC'].mean()),
            'high_volume_keywords': len(df[df['Volume'] >= 1000]),
            'low_difficulty_keywords': len(df[df['Difficulty'] <= 30]),
            'high_cpc_keywords': len(df[df['CPC'] >= 2.0]),
            'intent_distribution': self._get_intent_distribution(df)
        }
        
        return insights
    
    def _get_intent_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of search intents"""
        intent_counts = {}
        
        for intents in df['Intents']:
            if pd.notna(intents) and intents:
                intent_list = [intent.strip() for intent in str(intents).split(',')]
                for intent in intent_list:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return intent_counts
    
    def save_file_metadata(self, file_data: Dict[str, Any]) -> str:
        """Save file metadata to database"""
        # This would typically save to database
        # For now, return a mock file ID
        import uuid
        return str(uuid.uuid4())
    
    def get_parsing_errors(self, file_path: str) -> List[str]:
        """Get parsing errors for a file"""
        errors = []
        
        try:
            # Check file exists
            if not Path(file_path).exists():
                errors.append("File does not exist")
                return errors
            
            # Check file size
            if not self.validate_file_size(file_path):
                errors.append(f"File too large (max {settings.max_file_size} bytes)")
            
            # Check file type
            if not self.validate_file_type(file_path):
                errors.append("Invalid file type (must be .tsv or .txt)")
            
            # Try to parse
            try:
                df = pd.read_csv(file_path, sep='\t', nrows=1)
                self._validate_columns(df)
            except Exception as e:
                errors.append(f"Invalid TSV format: {str(e)}")
            
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
        
        return errors

# Global instance
file_parser_service = FileParserService()

