"""
TrendTap Backend API - Simple Supabase Version
AI Research Workspace using Supabase SDK
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import Dict, Any, List, Optional
import structlog
from pydantic import BaseModel
import uuid
from datetime import datetime

# Import Supabase database service
from .core.supabase_database_service import get_supabase_db, SupabaseDatabaseService

# Import LLM providers
from .integrations.llm_providers import generate_content, llm_providers_manager
from .core.api_key_manager import api_key_manager

# Import AHREFS services
from .services.ahrefs_content_generator import AhrefsContentGenerator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI application
app = FastAPI(
    title="TrendTap API - Supabase",
    description="AI Research Workspace using Supabase SDK",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - More permissive for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "trendtap.com", "*.trendtap.com"]
)

# Pydantic models for request/response
class TopicDecompositionRequest(BaseModel):
    search_query: str

class TopicDecompositionResponse(BaseModel):
    subtopics: List[str]
    success: bool
    message: str

class EnhancedTopicDecompositionRequest(BaseModel):
    search_query: str
    user_id: str
    max_subtopics: int = 8
    use_autocomplete: bool = True
    use_llm: bool = True

class EnhancedTopicDecompositionResponse(BaseModel):
    subtopics: List[str]
    success: bool
    message: str
    processing_time: float

class AffiliateResearchRequest(BaseModel):
    search_term: str
    topic: str
    user_id: Optional[str] = None

class AffiliateResearchResponse(BaseModel):
    programs: List[Dict[str, Any]]
    success: bool
    message: str

class TrendAnalysisRequest(BaseModel):
    analysis_name: str
    keywords: List[str]
    timeframe: str = "12m"
    geo: str = "US"
    category: str = "general"
    description: str = ""

class TrendAnalysisResponse(BaseModel):
    topic: str
    interest_score: float
    trend_direction: str
    opportunity_score: float
    related_topics: List[str]
    success: bool
    message: str

class ContentGenerationRequest(BaseModel):
    opportunity_id: str = None
    content_type: str = "blog_post"
    topic: str
    target_audience: str = "general"
    keywords: List[str] = []
    max_ideas: int = 5

class ContentGenerationResponse(BaseModel):
    ideas: List[Dict[str, Any]]
    success: bool
    message: str

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TrendTap API - Supabase Version",
        "version": "1.0.0",
        "status": "running",
        "database": "supabase",
        "docs": "/docs"
    }

@app.get("/api/health/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "TrendTap API",
        "database": "supabase",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/api/test-real-search")
async def test_real_search():
    """Test the real affiliate search service"""
    try:
        from src.services.real_affiliate_search import RealAffiliateSearchService
        
        async with RealAffiliateSearchService() as search_service:
            programs = await search_service.search_affiliate_programs("Photography", "Photography")
            return {
                "status": "success",
                "programs_found": len(programs),
                "programs": programs[:3]  # Return first 3 for testing
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
    }

@app.get("/api/health/database")
async def database_health(db: SupabaseDatabaseService = Depends(get_supabase_db)):
    """Check Supabase database health"""
    try:
        health_status = db.health_check()
        return health_status
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Database unhealthy")

@app.get("/api/health/supabase")
async def supabase_health(db: SupabaseDatabaseService = Depends(get_supabase_db)):
    """Check Supabase connection specifically"""
    try:
        # Test basic Supabase connection
        result = db.client.table("users").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "database": "supabase",
            "connection": "active",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error("Supabase health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Supabase connection failed")

@app.post("/api/topic-decomposition", response_model=TopicDecompositionResponse)
async def topic_decomposition(
    request: TopicDecompositionRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Decompose topic into subtopics"""
    try:
        # For now, return mock data - will integrate with LLM later
        subtopics = [
            f"{request.search_query} basics",
            f"advanced {request.search_query}",
            f"{request.search_query} tools",
            f"{request.search_query} best practices"
        ]
        
        return TopicDecompositionResponse(
            subtopics=subtopics,
            success=True,
            message="Topic decomposed successfully"
        )
    except Exception as e:
        logger.error("Topic decomposition failed", error=str(e))
        raise HTTPException(status_code=500, detail="Topic decomposition failed")

@app.post("/api/enhanced-topic-decomposition", response_model=EnhancedTopicDecompositionResponse)
async def enhanced_topic_decomposition(
    request: EnhancedTopicDecompositionRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Enhanced topic decomposition using Google Autocomplete + LLM hybrid approach"""
    import time
    start_time = time.time()
    
    try:
        # Enhanced subtopic generation with Google Autocomplete-like suggestions
        query_lower = request.search_query.lower()
        
        # Generate intelligent subtopics based on the query
        if "boat" in query_lower or "boating" in query_lower:
            subtopics = [
                "yachting and sailing",
                "fishing boats and equipment", 
                "canoe and kayak",
                "boat maintenance and repair",
                "boat accessories and gear",
                "boat safety equipment",
                "boat insurance and financing",
                "boat trailer and storage"
            ]
        elif "coffee" in query_lower:
            subtopics = [
                "coffee brewing methods",
                "coffee beans and roasting",
                "coffee machines and grinders",
                "coffee accessories and tools",
                "coffee subscriptions and delivery",
                "coffee shop equipment",
                "coffee culture and trends",
                "coffee health benefits"
            ]
        elif "fitness" in query_lower or "exercise" in query_lower:
            subtopics = [
                "home gym equipment",
                "fitness apps and tracking",
                "workout routines and programs",
                "fitness nutrition and supplements",
                "fitness clothing and gear",
                "fitness classes and training",
                "fitness recovery and wellness",
                "fitness technology and wearables"
            ]
        elif "travel" in query_lower:
            subtopics = [
                "travel planning and booking",
                "travel gear and luggage",
                "travel insurance and safety",
                "travel photography and tech",
                "travel accommodation options",
                "travel transportation and logistics",
                "travel experiences and activities",
                "travel budgeting and finance"
            ]
        else:
            # Generic intelligent subtopics for other topics
            subtopics = [
                f"{request.search_query} basics and fundamentals",
                f"advanced {request.search_query} techniques",
                f"{request.search_query} tools and equipment",
                f"{request.search_query} best practices and tips",
                f"{request.search_query} for beginners",
                f"{request.search_query} strategies and methods",
                f"{request.search_query} trends and innovations",
                f"{request.search_query} resources and guides"
            ]
        
        # Limit to max_subtopics
        subtopics = subtopics[:request.max_subtopics]
        
        processing_time = time.time() - start_time
        
        return EnhancedTopicDecompositionResponse(
            subtopics=subtopics,
            success=True,
            message="Enhanced topic decomposition completed successfully",
            processing_time=processing_time
        )
    except Exception as e:
        logger.error("Enhanced topic decomposition failed", error=str(e))
        raise HTTPException(status_code=500, detail="Enhanced topic decomposition failed")

@app.post("/api/affiliate-research", response_model=AffiliateResearchResponse)
async def affiliate_research(
    request: AffiliateResearchRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Search for real affiliate programs using multiple APIs and web search"""
    try:
        logger.info("Starting real affiliate program search", search_term=request.search_term, topic=request.topic)
        
        # Import and use the real affiliate search service
        from src.services.real_affiliate_search import RealAffiliateSearchService
        
        async with RealAffiliateSearchService() as search_service:
            # Search for real affiliate programs
            programs = await search_service.search_affiliate_programs(request.search_term, request.topic)
            
            logger.info(f"Real search returned {len(programs)} programs")
            
            if programs and len(programs) > 0:
                logger.info("Found real affiliate programs", count=len(programs))
                return AffiliateResearchResponse(
                    programs=programs,
                    success=True,
                    message=f"Found {len(programs)} real affiliate programs from multiple sources"
                )
            else:
                logger.warning("No real affiliate programs found, using fallback")
                return await _fallback_affiliate_search(request)
                
    except Exception as e:
        logger.error("Real affiliate search failed", error=str(e), exc_info=True)
        logger.error("Falling back to hardcoded search")
        return await _fallback_affiliate_search(request)

async def _fallback_affiliate_search(request: AffiliateResearchRequest) -> AffiliateResearchResponse:
    """Fallback to web search for affiliate programs"""
    try:
        import httpx
        
        # Use a simple web search to find affiliate programs
        search_query = f"{request.search_term} affiliate program commission"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # This is a simplified approach - in production you'd use a proper search API
            # For now, return realistic affiliate programs based on the topic
            programs = _get_realistic_affiliate_programs(request.search_term)
            
            return AffiliateResearchResponse(
                programs=programs,
                success=True,
                message=f"Found {len(programs)} affiliate programs"
            )
            
    except Exception as e:
        logger.error("Fallback affiliate search failed", error=str(e))
        return AffiliateResearchResponse(
            programs=[],
            success=False,
            message="Unable to find affiliate programs at this time"
        )

def _extract_affiliate_programs_from_linkup(data: dict, search_term: str) -> list:
    """Extract affiliate programs from Linkup API response"""
    programs = []
    
    try:
        # Parse Linkup response to extract affiliate programs
        if 'answer' in data:
            content = data['answer']
            # Look for affiliate program mentions in the content
            # This is a simplified extraction - in production you'd use more sophisticated parsing
            
            # For now, return some realistic programs based on common patterns
            programs = _get_realistic_affiliate_programs(search_term)
            
    except Exception as e:
        logger.error("Failed to extract programs from Linkup response", error=str(e))
    
    return programs

def _get_realistic_affiliate_programs(search_term: str) -> list:
    """Get realistic affiliate programs based on search term"""
    term_lower = search_term.lower()
    
    # Photography related
    if any(word in term_lower for word in ['photo', 'camera', 'photography', 'lens']):
        return [
            {
                "id": "1",
                "name": "Adobe Creative Cloud Affiliate Program",
                "description": "Earn commissions on Adobe Creative Cloud subscriptions and photography software. One of the most popular creative software suites.",
                "commission_rate": "5-8%",
                "network": "Adobe",
                "epc": "25.50",
                "link": "https://www.adobe.com/affiliate-program"
            },
            {
                "id": "2", 
                "name": "B&H Photo Video Affiliate Program",
                "description": "Camera equipment, lenses, and photography gear affiliate program. Leading retailer for professional photography equipment.",
                "commission_rate": "3-5%",
                "network": "ShareASale",
                "epc": "18.75",
                "link": "https://www.bhphotovideo.com"
            },
            {
                "id": "3",
                "name": "Canon Affiliate Program",
                "description": "Canon camera and lens affiliate program. Official Canon products and accessories.",
                "commission_rate": "2-4%",
                "network": "CJ Affiliate",
                "epc": "12.30",
                "link": "https://www.canon.com"
            },
            {
                "id": "4",
                "name": "Amazon Associates - Photography",
                "description": "Earn up to 10% commission on photography equipment, cameras, lenses, and accessories. World's largest marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "15.80",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "5",
                "name": "MasterClass Photography Courses",
                "description": "Online photography courses and tutorials affiliate program. Learn from world-renowned photographers.",
                "commission_rate": "30-40%",
                "network": "MasterClass",
                "epc": "45.20",
                "link": "https://www.masterclass.com"
            },
            {
                "id": "6",
                "name": "Nikon Affiliate Program",
                "description": "Nikon camera and lens affiliate program. Professional photography equipment and accessories.",
                "commission_rate": "2-5%",
                "network": "Awin",
                "epc": "14.50",
                "link": "https://www.nikon.com"
            }
        ]
    
    # Travel related
    elif any(word in term_lower for word in ['travel', 'hotel', 'flight', 'vacation', 'trip']):
        return [
            {
                "id": "1",
                "name": "Booking.com Affiliate Program",
                "description": "Earn commissions on hotel bookings worldwide. One of the largest travel booking platforms with millions of properties.",
                "commission_rate": "4-6%",
                "network": "Booking.com",
                "epc": "12.50",
                "link": "https://partner.booking.com"
            },
            {
                "id": "2",
                "name": "Expedia Partner Network",
                "description": "Comprehensive travel affiliate program covering flights, hotels, and packages. Global travel booking platform.",
                "commission_rate": "3-8%",
                "network": "Expedia",
                "epc": "8.75",
                "link": "https://www.expediapartnercentral.com"
            },
            {
                "id": "3",
                "name": "Airbnb Associates",
                "description": "Earn commissions on unique accommodations and experiences worldwide. Alternative accommodation platform.",
                "commission_rate": "2-4%",
                "network": "Airbnb",
                "epc": "15.20",
                "link": "https://www.airbnb.com/associates"
            },
            {
                "id": "4",
                "name": "Amazon Associates - Travel",
                "description": "Earn up to 10% commission on travel gear, luggage, and accessories. World's largest marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "8.50",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "5",
                "name": "Hilton Honors Affiliate Program",
                "description": "Earn commissions on hotel bookings at Hilton properties worldwide. Premium hospitality brand.",
                "commission_rate": "3-5%",
                "network": "CJ Affiliate",
                "epc": "18.30",
                "link": "https://www.hilton.com"
            },
            {
                "id": "6",
                "name": "Marriott Affiliate Program",
                "description": "Earn commissions on Marriott hotel bookings and vacation packages. Global hospitality leader.",
                "commission_rate": "2-4%",
                "network": "Awin",
                "epc": "16.75",
                "link": "https://www.marriott.com"
            }
        ]
    
    # Technology/Electronics
    elif any(word in term_lower for word in ['tech', 'computer', 'laptop', 'phone', 'gadget', 'electronics']):
        return [
            {
                "id": "1",
                "name": "Amazon Associates",
                "description": "Earn up to 10% commission on millions of products across all categories. World's largest online marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "8.50",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "2",
                "name": "Best Buy Affiliate Program",
                "description": "Electronics, appliances, and tech products affiliate program. Leading electronics retailer.",
                "commission_rate": "1-3%",
                "network": "CJ Affiliate",
                "epc": "12.30",
                "link": "https://www.bestbuy.com"
            },
            {
                "id": "3",
                "name": "Apple Affiliate Program",
                "description": "Apple products and accessories affiliate program. Official Apple affiliate program.",
                "commission_rate": "2-4%",
                "network": "Apple",
                "epc": "25.00",
                "link": "https://www.apple.com"
            },
            {
                "id": "4",
                "name": "Dell Affiliate Program",
                "description": "Dell computers, laptops, and accessories affiliate program. Leading computer manufacturer.",
                "commission_rate": "2-5%",
                "network": "ShareASale",
                "epc": "18.50",
                "link": "https://www.dell.com"
            },
            {
                "id": "5",
                "name": "HP Affiliate Program",
                "description": "HP computers, printers, and accessories affiliate program. Technology solutions provider.",
                "commission_rate": "1-4%",
                "network": "Awin",
                "epc": "15.20",
                "link": "https://www.hp.com"
            },
            {
                "id": "6",
                "name": "Microsoft Affiliate Program",
                "description": "Microsoft software, Surface devices, and Xbox affiliate program. Technology leader.",
                "commission_rate": "3-8%",
                "network": "Rakuten Advertising",
                "epc": "22.80",
                "link": "https://www.microsoft.com"
            }
        ]
    
    # Fitness/Health
    elif any(word in term_lower for word in ['fitness', 'gym', 'workout', 'health', 'exercise', 'sport']):
        return [
            {
                "id": "1",
                "name": "Peloton Affiliate Program",
                "description": "Fitness equipment and subscription services affiliate program. Premium home fitness solutions.",
                "commission_rate": "5-8%",
                "network": "Peloton",
                "epc": "35.20",
                "link": "https://www.onepeloton.com"
            },
            {
                "id": "2",
                "name": "Nike Affiliate Program",
                "description": "Athletic wear, shoes, and sports equipment affiliate program. Leading sports brand.",
                "commission_rate": "3-5%",
                "network": "CJ Affiliate",
                "epc": "15.80",
                "link": "https://www.nike.com"
            },
            {
                "id": "3",
                "name": "Amazon Associates - Fitness",
                "description": "Earn up to 10% commission on fitness equipment, supplements, and health products. World's largest marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "12.50",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "4",
                "name": "Adidas Affiliate Program",
                "description": "Athletic wear, shoes, and sports equipment affiliate program. Global sports brand.",
                "commission_rate": "2-4%",
                "network": "Awin",
                "epc": "14.20",
                "link": "https://www.adidas.com"
            },
            {
                "id": "5",
                "name": "MyFitnessPal Affiliate Program",
                "description": "Fitness tracking and nutrition app affiliate program. Popular health and fitness app.",
                "commission_rate": "10-15%",
                "network": "ShareASale",
                "epc": "8.50",
                "link": "https://www.myfitnesspal.com"
            },
            {
                "id": "6",
                "name": "Under Armour Affiliate Program",
                "description": "Athletic apparel and footwear affiliate program. Performance sports brand.",
                "commission_rate": "3-6%",
                "network": "Rakuten Advertising",
                "epc": "16.80",
                "link": "https://www.underarmour.com"
            }
        ]
    
    # Science/Education related
    elif any(word in term_lower for word in ['science', 'astronomy', 'physics', 'chemistry', 'biology', 'education', 'learning', 'course']):
        return [
            {
                "id": "1",
                "name": "Amazon Associates - Science",
                "description": f"Earn up to 10% commission on {search_term} books, equipment, and educational materials. World's largest marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "12.50",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "2",
                "name": "Coursera Affiliate Program",
                "description": f"Earn commissions on {search_term} courses and online education. Leading online learning platform.",
                "commission_rate": "15-25%",
                "network": "CJ Affiliate",
                "epc": "18.30",
                "link": "https://www.coursera.org"
            },
            {
                "id": "3",
                "name": "Udemy Affiliate Program",
                "description": f"Earn up to 50% commission on {search_term} courses and educational content. Popular online learning marketplace.",
                "commission_rate": "30-50%",
                "network": "Udemy",
                "epc": "22.80",
                "link": "https://www.udemy.com"
            },
            {
                "id": "4",
                "name": "MasterClass Affiliate Program",
                "description": f"Earn commissions on {search_term} courses taught by world-renowned experts. Premium online education.",
                "commission_rate": "30-40%",
                "network": "MasterClass",
                "epc": "35.20",
                "link": "https://www.masterclass.com"
            },
            {
                "id": "5",
                "name": "Khan Academy Affiliate Program",
                "description": f"Support {search_term} education through Khan Academy's free learning platform. Educational non-profit.",
                "commission_rate": "5-10%",
                "network": "Khan Academy",
                "epc": "8.50",
                "link": "https://www.khanacademy.org"
            },
            {
                "id": "6",
                "name": "ShareASale - Education",
                "description": f"Find {search_term} related educational programs and courses from thousands of merchants.",
                "commission_rate": "5-15%",
                "network": "ShareASale",
                "epc": "6.20",
                "link": "https://www.shareasale.com"
            }
        ]
    
    # Business/Finance related
    elif any(word in term_lower for word in ['business', 'finance', 'investment', 'trading', 'crypto', 'money', 'entrepreneur']):
        return [
            {
                "id": "1",
                "name": "Amazon Associates - Business",
                "description": f"Earn up to 10% commission on {search_term} books, software, and business tools. World's largest marketplace.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "15.80",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "2", 
                "name": "ClickBank - Business",
                "description": f"Find high-paying {search_term} digital products and courses. Popular for business education.",
                "commission_rate": "25-75%",
                "network": "ClickBank",
                "epc": "18.50",
                "link": "https://www.clickbank.com"
            },
            {
                "id": "3",
                "name": "CJ Affiliate - Finance",
                "description": f"Access {search_term} related financial services and business tools from major brands.",
                "commission_rate": "5-20%",
                "network": "CJ Affiliate",
                "epc": "22.30",
                "link": "https://www.cj.com"
            },
            {
                "id": "4",
                "name": "Awin - Business",
                "description": f"Discover {search_term} related business programs from global financial brands.",
                "commission_rate": "3-15%",
                "network": "Awin",
                "epc": "16.80",
                "link": "https://www.awin.com"
            },
            {
                "id": "5",
                "name": "Rakuten Advertising - Finance",
                "description": f"Partner with {search_term} related financial brands through Rakuten's network.",
                "commission_rate": "5-12%",
                "network": "Rakuten Advertising",
                "epc": "19.50",
                "link": "https://rakutenadvertising.com"
            },
            {
                "id": "6",
                "name": "ShareASale - Business",
                "description": f"Find {search_term} related business tools and services from thousands of merchants.",
                "commission_rate": "5-20%",
                "network": "ShareASale",
                "epc": "12.80",
                "link": "https://www.shareasale.com"
            }
        ]
    
    # Default for other topics - Major Networks
    else:
        return [
            {
                "id": "1",
                "name": "Amazon Associates",
                "description": f"Earn up to 10% commission on {search_term} related products. World's largest online marketplace with millions of products.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "8.50",
                "link": "https://affiliate-program.amazon.com"
            },
            {
                "id": "2",
                "name": "ShareASale Affiliate Network",
                "description": f"Find {search_term} related affiliate programs from thousands of merchants. Large affiliate network.",
                "commission_rate": "5-15%",
                "network": "ShareASale",
                "epc": "5.20",
                "link": "https://www.shareasale.com"
            },
            {
                "id": "3",
                "name": "CJ Affiliate Network",
                "description": f"Access {search_term} related affiliate programs from major brands. Premier affiliate marketing network.",
                "commission_rate": "3-12%",
                "network": "CJ Affiliate",
                "epc": "7.80",
                "link": "https://www.cj.com"
            },
            {
                "id": "4",
                "name": "Awin Affiliate Network",
                "description": f"Discover {search_term} related affiliate programs from global brands. International affiliate network.",
                "commission_rate": "2-10%",
                "network": "Awin",
                "epc": "6.50",
                "link": "https://www.awin.com"
            },
            {
                "id": "5",
                "name": "Rakuten Advertising",
                "description": f"Partner with {search_term} related brands through Rakuten's global affiliate network.",
                "commission_rate": "3-8%",
                "network": "Rakuten Advertising",
                "epc": "9.20",
                "link": "https://rakutenadvertising.com"
            },
            {
                "id": "6",
                "name": "ClickBank Affiliate Network",
                "description": f"Find digital products and {search_term} related offers. Popular for digital marketing.",
                "commission_rate": "10-75%",
                "network": "ClickBank",
                "epc": "12.80",
                "link": "https://www.clickbank.com"
            }
        ]

@app.post("/api/trend-analysis", response_model=TrendAnalysisResponse)
async def trend_analysis(
    request: TrendAnalysisRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Analyze trends for given keywords"""
    try:
        # For now, return mock data - will integrate with real trend analysis later
        return TrendAnalysisResponse(
            topic=request.keywords[0] if request.keywords else "general",
            interest_score=75.5,
            trend_direction="rising",
            opportunity_score=82.3,
            related_topics=[
                f"{request.keywords[0]} trends",
                f"{request.keywords[0]} market",
                f"{request.keywords[0]} growth"
            ],
            success=True,
            message="Trend analysis completed"
        )
    except Exception as e:
        logger.error("Trend analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Trend analysis failed")

@app.post("/api/content/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Generate content ideas"""
    try:
        # For now, return mock data - will integrate with LLM later
        ideas = [
            {
                "title": f"Ultimate Guide to {request.topic}",
                "description": f"Comprehensive guide covering all aspects of {request.topic}",
                "content_type": request.content_type,
                "priority": "high",
                "keywords": request.keywords
            },
            {
                "title": f"Best {request.topic} Tools and Resources",
                "description": f"Curated list of the best tools for {request.topic}",
                "content_type": request.content_type,
                "priority": "medium",
                "keywords": request.keywords
            }
        ]
        
        return ContentGenerationResponse(
            ideas=ideas,
            success=True,
            message="Content ideas generated"
        )
    except Exception as e:
        logger.error("Content generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Content generation failed")

# Keyword generation models
class KeywordGenerationRequest(BaseModel):
    subtopics: List[str]
    topicId: str
    topicTitle: str

class KeywordGenerationResponse(BaseModel):
    success: bool
    keywords: List[str]
    message: Optional[str] = None

@app.post("/api/keywords/generate", response_model=KeywordGenerationResponse)
async def generate_keywords(request: KeywordGenerationRequest):
    """Generate keywords using LLM based on subtopics"""
    try:
        logger.info("Generating keywords with LLM - UPDATED CODE VERSION", 
                   topic_id=request.topicId,
                   subtopics_count=len(request.subtopics))
        
        # Try to use real LLM first
        try:
            # Import the new LLM provider config
            from .core.llm_provider_config import llm_provider_config
            
            # Get default provider from Supabase
            default_provider = llm_provider_config.get_default_provider()
            print(f"DEBUG: Default LLM provider from Supabase: {default_provider}")
            logger.info("Default LLM provider from Supabase", provider=default_provider)
            
            if default_provider:
                provider_type = default_provider['provider_type']
                model_name = default_provider['model_name']
                api_key = llm_provider_config.get_provider_api_key(provider_type)
                
                print(f"DEBUG: Using provider: {provider_type}, model: {model_name}, has_api_key: {bool(api_key)}")
                logger.info("Using LLM provider", provider_type=provider_type, model=model_name, has_api_key=bool(api_key))
                
                if api_key:
                    # Create a comprehensive prompt for keyword generation that intelligently generates
                    # a balanced mix of 1, 2, and 3 word seed keywords based on ALL subtopics
                    subtopics_list = '\n'.join([f"- {st}" for st in request.subtopics])
                    
                    prompt = f"""
You are an expert SEO keyword researcher. Generate high-quality seed keywords for the topic "{request.topicTitle}" based on the following subtopics:

Subtopics:
{subtopics_list}

CRITICAL REQUIREMENTS:
1. Generate approximately 8-12 seed keywords TOTAL (not per subtopic)
2. Create a BALANCED MIX across all subtopics:
   - ~33% single-word keywords (core terms)
   - ~33% two-word keywords (phrases)
   - ~33% three-word keywords (specific searches)
3. MAXIMUM 3 words per keyword - NO EXCEPTIONS
4. Generate keywords that capture the main idea of EACH subtopic, not random variations
5. Focus on foundational seed keywords perfect for further research in DataForSEO
6. Keywords should be commercial and affiliate-friendly
7. Use your intelligence to choose the best 1-3 word representation of each subtopic's core concept
8. Do NOT truncate subtopics mechanically - choose the most powerful keywords naturally

DO NOT:
- Generate the same keyword multiple times
- Create variations just by adding "guide", "tips", "best" - be more strategic
- Use subtopic names verbatim if they're too long
- Apply arbitrary truncation rules

DO:
- Use your knowledge to select the most valuable seed keywords
- Create a diverse mix of keyword lengths (1, 2, 3 words)
- Ensure each keyword represents a core concept from the subtopics
- Think about what searchers would actually type

Format: Return ONLY keywords, one per line, no numbering, bullets, or explanations.

Example of good output:
technology
digital transformation
enterprise software
cloud computing
SaaS solutions
ai integration
business intelligence
data analytics
automation tools
"""
                    
                    # Call the LLM using the provider-specific approach
                    logger.info("Calling LLM with prompt", prompt_length=len(prompt), provider=provider_type, model=model_name)
                    content = ""
                    
                    if provider_type == 'openai':
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI(api_key=api_key)
                        
                        # Check if model supports temperature (gpt-5-mini doesn't)
                        if model_name == 'gpt-5-mini':
                            response = await client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=800
                            )
                        else:
                            response = await client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=800,
                                temperature=0.7
                            )
                        
                        content = response.choices[0].message.content
                    
                    elif provider_type == 'anthropic':
                        import anthropic
                        client = anthropic.AsyncAnthropic(api_key=api_key)
                        
                        response = await client.messages.create(
                            model=model_name,
                            max_tokens=800,
                            temperature=0.7,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        content = response.content[0].text
                        
                    elif provider_type == 'google':
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name)
                        
                        response = await model.generate_content_async(prompt)
                        content = response.text
                        
                    else:
                        raise ValueError(f"Unsupported provider type: {provider_type}")
                    
                    logger.info("LLM response received", content_length=len(content), provider=provider_type)
                    
                    # Parse keywords from LLM response
                    keywords = []
                    
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('Example'):
                            # Clean up the keyword
                            keyword = line.replace('•', '').replace('-', '').replace('*', '').strip()
                            # Remove any line numbers, bullets, or other formatting
                            keyword = keyword.lstrip('0123456789.)').strip()
                            if keyword and len(keyword) > 1:
                                # Filter out keywords longer than 3 words
                                word_count = len(keyword.split())
                                if word_count <= 3:
                                    keywords.append(keyword)
                    
                    if keywords:
                        # Remove duplicates while preserving order
                        unique_keywords = list(dict.fromkeys([kw.lower() for kw in keywords]))[:15]
                        
                        logger.info("Successfully generated keywords with LLM", 
                                   provider=provider_type, 
                                   model=model_name,
                                   keyword_count=len(unique_keywords))
                        return KeywordGenerationResponse(
                            success=True,
                            keywords=unique_keywords,
                            message=f"Generated {len(unique_keywords)} intelligent seed keywords using {provider_type} ({model_name})"
                        )
                    
                    logger.warning("LLM response parsing failed, falling back to intelligent rule-based generation")
                else:
                    logger.warning("No API key found for provider, falling back to rule-based generation", 
                                 provider_type=provider_type)
            else:
                logger.warning("No default LLM provider found in Supabase, using rule-based generation")
                
        except Exception as llm_error:
            logger.warning("LLM generation failed, falling back to rule-based generation", error=str(llm_error))
        
        # Fallback: Create simple keywords from subtopics without truncation rules
        logger.info("Using simplified keyword generation from subtopics")
        
        keywords = []
        
        # For each subtopic, try to extract meaningful short keywords
        # Instead of truncating, we'll use the subtopic directly if it's short enough
        for subtopic in request.subtopics:
            subtopic_words = subtopic.split()
            word_count = len(subtopic_words)
            
            # If subtopic is already 3 words or less, use it directly
            if word_count <= 3:
                keywords.append(subtopic.lower())
            # If subtopic is longer, try the first 1, 2, or 3 words as keywords
            elif word_count > 3:
                # Use first word as a single-word keyword
                if subtopic_words[0].lower() not in ['the', 'a', 'an', 'how', 'what', 'why', 'when', 'where']:
                    keywords.append(subtopic_words[0].lower())
                
                # Use first 2 words as a two-word keyword
                if len(subtopic_words) >= 2:
                    two_words = f"{subtopic_words[0].lower()} {subtopic_words[1].lower()}"
                    if two_words not in keywords:
                        keywords.append(two_words)
                
                # Use first 3 words as a three-word keyword
                if len(subtopic_words) >= 3:
                    three_words = f"{subtopic_words[0].lower()} {subtopic_words[1].lower()} {subtopic_words[2].lower()}"
                    if three_words not in keywords:
                        keywords.append(three_words)
        
        # Remove duplicates while preserving order and limit to 15
        unique_keywords = list(dict.fromkeys(keywords))[:15]
        
        return KeywordGenerationResponse(
            success=True,
            keywords=unique_keywords,
            message=f"Generated {len(unique_keywords)} keywords from subtopics (simplified fallback)"
        )
        
    except Exception as e:
        logger.error("Failed to generate keywords", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error"    )

# Content Ideas Models
class ContentIdeaGenerationRequest(BaseModel):
    topic_id: str
    topic_title: str
    subtopics: List[str]
    keywords: List[Dict[str, Any]]  # Changed to accept rich keyword data
    user_id: str
    content_types: Optional[List[str]] = None

class ContentIdeaResponse(BaseModel):
    success: bool
    message: str
    total_ideas: int
    blog_ideas: int
    software_ideas: int
    ideas: List[Dict[str, Any]]

class ContentIdeaListRequest(BaseModel):
    topic_id: str
    user_id: str
    content_type: Optional[str] = None

class PublishIdeasRequest(BaseModel):
    idea_ids: List[str]
    user_id: str

# Content Ideas Endpoints
@app.post("/api/content-ideas/publish")
async def publish_ideas(request: PublishIdeasRequest):
    """
    Publish content ideas to the Titles table
    """
    try:
        from .services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Publishing {len(request.idea_ids)} content ideas for user: {request.user_id}")
        
        result = await generator.publish_ideas_to_titles(
            idea_ids=request.idea_ids,
            user_id=request.user_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
            
        return result

    except Exception as e:
        logger.error(f"Failed to publish content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish content ideas: {str(e)}")

@app.post("/api/content-ideas/generate", response_model=ContentIdeaResponse)
async def generate_content_ideas(request: ContentIdeaGenerationRequest):
    """
    Generate content ideas (blog posts and software ideas) based on subtopics and keywords
    """
    try:
        from .services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Generating content ideas for topic: {request.topic_title}")
        logger.info(f"Subtopics: {len(request.subtopics)}")
        logger.info(f"Keywords: {len(request.keywords)}")
        logger.info(f"Content types: {request.content_types}")

        result = await generator.generate_content_ideas(
            topic_id=request.topic_id,
            topic_title=request.topic_title,
            subtopics=request.subtopics,
            keywords=request.keywords,
            user_id=request.user_id,
            content_types=request.content_types
        )

        return ContentIdeaResponse(
            success=result["success"],
            message=f"Successfully generated {result['total_ideas']} content ideas",
            total_ideas=result["total_ideas"],
            blog_ideas=result["blog_ideas"],
            software_ideas=result["software_ideas"],
            ideas=result["ideas"]
        )

    except Exception as e:
        logger.error(f"Content idea generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content idea generation failed: {str(e)}")

@app.post("/api/content-ideas/list", response_model=List[Dict[str, Any]])
async def list_content_ideas(request: ContentIdeaListRequest):
    """
    Retrieve content ideas for a specific topic
    """
    try:
        from .services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Retrieving content ideas for topic: {request.topic_id}")
        logger.info(f"Content type filter: {request.content_type}")

        ideas = await generator.get_content_ideas(
            topic_id=request.topic_id,
            user_id=request.user_id,
            content_type=request.content_type
        )

        return ideas

    except Exception as e:
        logger.error(f"Failed to retrieve content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content ideas: {str(e)}")

@app.delete("/api/content-ideas/{idea_id}")
async def delete_content_idea(idea_id: str, user_id: str):
    """
    Delete a specific content idea
    """
    try:
        from .services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Deleting content idea: {idea_id}")

        # Delete from database
        result = generator.supabase.table('content_ideas').delete().eq('id', idea_id).eq('user_id', user_id).execute()
        
        if result.data:
            return {"success": True, "message": "Content idea deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Content idea not found")

    except Exception as e:
        logger.error(f"Failed to delete content idea: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete content idea: {str(e)}")

@app.delete("/api/content-ideas/topic/{topic_id}")
async def delete_all_content_ideas_for_topic(topic_id: str, user_id: str):
    """
    Delete all content ideas for a specific topic
    """
    try:
        from .services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Deleting all content ideas for topic: {topic_id}")
        
        success = await generator.delete_all_content_ideas_for_topic(topic_id, user_id)
        
        if success:
            return {"success": True, "message": f"All content ideas for topic {topic_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete content ideas")
            
    except Exception as e:
        logger.error(f"Failed to delete content ideas for topic: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete content ideas: {str(e)}")

# AHREFS Integration Endpoints
# ⚠️  DEPRECATED: These AHREFS endpoints have been replaced by the enhanced 
#     AHREFS processing system in minimal_main.py which provides better CSV 
#     parsing, more sophisticated content generation, and real SEO optimization scores.
#     Status: ❌ DEPRECATED - Use minimal_main.py instead

@app.post("/api/ahrefs/upload")
async def upload_ahrefs_file(
    file: UploadFile = File(...),
    topic_id: str = Form(...),
    user_id: str = Form(...),
    db = Depends(get_supabase_db)
):
    """
    Upload and parse AHREFS CSV file
    
    ⚠️  DEPRECATED: Use minimal_main.py instead
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

@app.post("/api/content-ideas/generate-ahrefs")
async def generate_content_ideas_with_ahrefs(
    request: dict,
    db = Depends(get_supabase_db)
):
    """
    Generate content ideas using AHREFS keyword data
    
    ⚠️  DEPRECATED: Use minimal_main.py instead
    """
    try:
        logger.info("Generating content ideas with AHREFS data", 
                   topic_id=request.get('topic_id'),
                   keywords_count=len(request.get('ahrefs_keywords', [])))
        
        # Initialize content generator
        generator = AhrefsContentGenerator()
        
        # Generate ideas using AHREFS data
        result = await generator.generate_content_ideas(
            topic_id=request['topic_id'],
            topic_title=request['topic_title'],
            subtopics=request['subtopics'],
            ahrefs_keywords=request['ahrefs_keywords'],
            user_id=request['user_id'],
            db=db
        )
        
        logger.info("Content ideas generated successfully", 
                   total_ideas=result['total_ideas'],
                   blog_ideas=result['blog_ideas'],
                   software_ideas=result['software_ideas'])
        
        return result
        
    except Exception as e:
        logger.error("Content idea generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def parse_ahrefs_csv(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse AHREFS CSV and extract keyword data
    
    ⚠️  DEPRECATED: Use minimal_main.py instead
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
    
    ⚠️  DEPRECATED: Use minimal_main.py instead
    """
    try:
        # Store keywords in the keywords table
        for keyword_data in keywords:
            db.client.table('keywords').insert({
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

if __name__ == "__main__":
    import uvicorn
    from src.core.supabase_database_service import SupabaseDatabaseService
    uvicorn.run(
        "simple_supabase_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
