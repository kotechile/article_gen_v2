"""
AHREFS integration routes for content idea generation
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Dict, Any, Optional
import structlog
from pydantic import BaseModel
import uuid
from datetime import datetime

from ..services.ahrefs_content_generator import AhrefsContentGenerator
from ..core.supabase_database_service import get_supabase_db

logger = structlog.get_logger()
router = APIRouter(prefix="/api/ahrefs", tags=["ahrefs"])

class AhrefsKeyword(BaseModel):
    keyword: str
    volume: int
    difficulty: int
    cpc: float
    traffic_potential: int
    intents: List[str]
    serp_features: List[str]
    parent_keyword: Optional[str] = None
    country: str = "us"
    global_volume: int
    global_traffic_potential: int
    first_seen: str
    last_update: str

class AhrefsContentIdeasRequest(BaseModel):
    topic_id: str
    topic_title: str
    subtopics: List[str]
    ahrefs_keywords: List[AhrefsKeyword]
    user_id: str

class AhrefsContentIdeasResponse(BaseModel):
    success: bool
    message: str
    total_ideas: int
    blog_ideas: int
    software_ideas: int
    ideas: List[Dict[str, Any]]
    analytics_summary: Dict[str, Any]

@router.post("/upload")
async def upload_ahrefs_file(
    file: UploadFile = File(...),
    topic_id: str = Form(...),
    user_id: str = Form(...),
    db = Depends(get_supabase_db)
):
    """
    Upload and parse AHREFS CSV file
    """
    try:
        logger.info("Processing AHREFS file upload", 
                   filename=file.filename, 
                   topic_id=topic_id,
                   user_id=user_id)
        
        # Read file content
        content = await file.read()
        csv_text = content.decode('utf-8')
        
        # Parse CSV and extract keywords
        keywords = parse_ahrefs_csv(csv_text)
        
        if not keywords:
            raise HTTPException(status_code=400, detail="No valid keywords found in CSV file")
        
        # Store keywords in database
        file_id = str(uuid.uuid4())
        await store_ahrefs_keywords(db, file_id, topic_id, user_id, keywords)
        
        logger.info("AHREFS file processed successfully", 
                   file_id=file_id, 
                   keywords_count=len(keywords))
        
        return {
            "success": True,
            "message": f"Successfully processed {len(keywords)} keywords",
            "file_id": file_id,
            "keywords_count": len(keywords),
            "keywords": keywords[:10]  # Return first 10 for preview
        }
        
    except Exception as e:
        logger.error("AHREFS file upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/generate-content-ideas", response_model=AhrefsContentIdeasResponse)
async def generate_content_ideas_with_ahrefs(
    request: AhrefsContentIdeasRequest,
    db = Depends(get_supabase_db)
):
    """
    Generate content ideas using AHREFS keyword data
    """
    try:
        logger.info("Generating content ideas with AHREFS data", 
                   topic_id=request.topic_id,
                   keywords_count=len(request.ahrefs_keywords))
        
        # Initialize content generator
        generator = AhrefsContentGenerator()
        
        # Generate ideas using AHREFS data
        result = await generator.generate_content_ideas(
            topic_id=request.topic_id,
            topic_title=request.topic_title,
            subtopics=request.subtopics,
            ahrefs_keywords=request.ahrefs_keywords,
            user_id=request.user_id,
            db=db
        )
        
        logger.info("Content ideas generated successfully", 
                   total_ideas=result['total_ideas'],
                   blog_ideas=result['blog_ideas'],
                   software_ideas=result['software_ideas'])
        
        return AhrefsContentIdeasResponse(**result)
        
    except Exception as e:
        logger.error("Content idea generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def parse_ahrefs_csv(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse AHREFS CSV and extract keyword data
    """
    lines = csv_text.split('\n')
    if len(lines) < 2:
        raise ValueError("CSV file must have at least a header row and one data row")
    
    headers = lines[0].split('\t')
    keywords = []
    
    # Map AHREFS columns
    column_mapping = {
        'keyword': ['keyword', 'query', 'search term'],
        'volume': ['volume'],
        'difficulty': ['difficulty'],
        'cpc': ['cpc'],
        'traffic_potential': ['traffic potential', 'traffic'],
        'intents': ['intents'],
        'serp_features': ['serp features'],
        'parent_keyword': ['parent keyword'],
        'country': ['country'],
        'global_volume': ['global volume'],
        'global_traffic_potential': ['global traffic potential'],
        'first_seen': ['first seen'],
        'last_update': ['last update']
    }
    
    def find_column_index(target_columns):
        for target in target_columns:
            for i, header in enumerate(headers):
                if target.lower() in header.lower().replace('"', ''):
                    return i
        return -1
    
    keyword_index = find_column_index(column_mapping['keyword'])
    if keyword_index == -1:
        raise ValueError("Keyword column not found in CSV")
    
    volume_index = find_column_index(column_mapping['volume'])
    difficulty_index = find_column_index(column_mapping['difficulty'])
    cpc_index = find_column_index(column_mapping['cpc'])
    traffic_index = find_column_index(column_mapping['traffic_potential'])
    intents_index = find_column_index(column_mapping['intents'])
    serp_features_index = find_column_index(column_mapping['serp_features'])
    parent_keyword_index = find_column_index(column_mapping['parent_keyword'])
    country_index = find_column_index(column_mapping['country'])
    global_volume_index = find_column_index(column_mapping['global_volume'])
    global_traffic_index = find_column_index(column_mapping['global_traffic_potential'])
    first_seen_index = find_column_index(column_mapping['first_seen'])
    last_update_index = find_column_index(column_mapping['last_update'])
    
    for i, line in enumerate(lines[1:], 1):
        columns = line.split('\t')
        if len(columns) <= keyword_index:
            continue
            
        keyword = columns[keyword_index].strip().replace('"', '')
        if not keyword:
            continue
            
        keyword_data = {
            'keyword': keyword,
            'volume': int(columns[volume_index]) if volume_index != -1 and columns[volume_index].strip() else 0,
            'difficulty': int(columns[difficulty_index]) if difficulty_index != -1 and columns[difficulty_index].strip() else 0,
            'cpc': float(columns[cpc_index]) if cpc_index != -1 and columns[cpc_index].strip() else 0.0,
            'traffic_potential': int(columns[traffic_index]) if traffic_index != -1 and columns[traffic_index].strip() else 0,
            'intents': columns[intents_index].split(',') if intents_index != -1 and columns[intents_index].strip() else [],
            'serp_features': columns[serp_features_index].split(',') if serp_features_index != -1 and columns[serp_features_index].strip() else [],
            'parent_keyword': columns[parent_keyword_index].strip().replace('"', '') if parent_keyword_index != -1 and columns[parent_keyword_index].strip() else None,
            'country': columns[country_index].strip().replace('"', '') if country_index != -1 and columns[country_index].strip() else 'us',
            'global_volume': int(columns[global_volume_index]) if global_volume_index != -1 and columns[global_volume_index].strip() else 0,
            'global_traffic_potential': int(columns[global_traffic_index]) if global_traffic_index != -1 and columns[global_traffic_index].strip() else 0,
            'first_seen': columns[first_seen_index].strip().replace('"', '') if first_seen_index != -1 and columns[first_seen_index].strip() else '',
            'last_update': columns[last_update_index].strip().replace('"', '') if last_update_index != -1 and columns[last_update_index].strip() else ''
        }
        
        # Clean up intents and serp_features
        keyword_data['intents'] = [intent.strip() for intent in keyword_data['intents'] if intent.strip()]
        keyword_data['serp_features'] = [feature.strip() for feature in keyword_data['serp_features'] if feature.strip()]
        
        keywords.append(keyword_data)
    
    return keywords

async def store_ahrefs_keywords(db, file_id: str, topic_id: str, user_id: str, keywords: List[Dict[str, Any]]):
    """
    Store AHREFS keywords in the database
    """
    try:
        # Store keywords in the keywords table
        for keyword_data in keywords:
            await db.table('keywords').insert({
                'id': str(uuid.uuid4()),
                'keyword': keyword_data['keyword'],
                'search_volume': keyword_data['volume'],
                'difficulty': keyword_data['difficulty'],
                'cpc': keyword_data['cpc'],
                'topic_id': topic_id,
                'user_id': user_id,
                'source': 'ahrefs',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).execute()
        
        logger.info("AHREFS keywords stored successfully", 
                   file_id=file_id, 
                   topic_id=topic_id,
                   keywords_count=len(keywords))
        
    except Exception as e:
        logger.error("Failed to store AHREFS keywords", error=str(e))
        raise
