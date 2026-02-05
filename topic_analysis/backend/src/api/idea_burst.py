"""
Idea Burst API endpoints for content idea management and selection
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from ..core.supabase_database_service import get_database_service
from ..services.content_idea_generator import ContentIdeaGenerator
from ..services.keyword_analyzer import KeywordAnalyzerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/idea-burst", tags=["idea-burst"])

# Initialize services
database_service = get_database_service()
content_idea_generator = ContentIdeaGenerator()
keyword_analyzer = KeywordAnalyzerService()

@router.post("/{file_id}/generate-ideas")
async def generate_ideas_with_ahrefs(
    file_id: str,
    num_ideas_per_subtopic: int = Query(10, ge=5, le=20, description="Number of ideas per sub-topic"),
    include_software_ideas: bool = Query(True, description="Include software ideas generation")
):
    """
    Generate content ideas using Ahrefs keywords organized by sub-topics
    Creates 10+ SEO-optimized article ideas per sub-topic + separate software ideas
    
    Args:
        file_id: ID of the uploaded Ahrefs file
        num_ideas_per_subtopic: Number of ideas to generate per sub-topic
        include_software_ideas: Whether to include software ideas generation
        
    Returns:
        Generated content ideas organized by sub-topics
    """
    try:
        # Get analyzed keywords from the file
        keywords = await database_service.get_keywords(file_id)
        if not keywords:
            raise HTTPException(
                status_code=404,
                detail="No analyzed keywords found for this file"
            )
        
        # Get content opportunities
        content_opportunities = await database_service.get_content_opportunities(file_id)
        
        # Generate article ideas using Ahrefs keywords
        article_ideas = content_idea_generator.generate_content_ideas(
            keywords=keywords,
            content_opportunities=content_opportunities,
            num_ideas_per_subtopic=num_ideas_per_subtopic
        )
        
        # Generate software ideas separately (not keyword-dependent)
        software_ideas = []
        if include_software_ideas:
            software_ideas = content_idea_generator.generate_software_ideas()
        
        # Save ideas to database
        all_ideas = article_ideas + software_ideas
        await database_service.save_seo_content_ideas(all_ideas)
        
        # Organize by sub-topics
        subtopic_organization = _organize_ideas_by_subtopic(article_ideas)
        
        logger.info(f"Generated {len(article_ideas)} article ideas and {len(software_ideas)} software ideas for file {file_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "article_ideas": article_ideas,
                "software_ideas": software_ideas,
                "subtopic_organization": subtopic_organization,
                "total_ideas": len(all_ideas),
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ideas for file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/{report_id}/ideas")
async def get_content_ideas(
    report_id: str,
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    priority: Optional[str] = Query(None, description="Filter by priority level"),
    sort_by: str = Query("combined_score", description="Sort by field"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page")
):
    """
    Get content ideas for a report
    
    Args:
        report_id: ID of the analysis report
        content_type: Filter by content type
        priority: Filter by priority level
        sort_by: Sort by field
        sort_order: Sort order
        page: Page number
        limit: Items per page
        
    Returns:
        List of content ideas with selection indicators
    """
    try:
        # Get content ideas from database
        content_ideas = database_service.get_content_ideas_by_report(report_id)
        
        if not content_ideas:
            raise HTTPException(
                status_code=404,
                detail="No content ideas found for this report"
            )
        
        # Apply filters
        filtered_ideas = _apply_idea_filters(content_ideas, content_type, priority)
        
        # Sort ideas
        sorted_ideas = _sort_ideas(filtered_ideas, sort_by, sort_order)
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_ideas = sorted_ideas[start_idx:end_idx]
        
        # Add selection indicators
        ideas_with_indicators = _add_selection_indicators(paginated_ideas)
        
        return JSONResponse(
            status_code=200,
            content={
                "ideas": ideas_with_indicators,
                "total": len(filtered_ideas),
                "page": page,
                "limit": limit,
                "has_more": end_idx < len(filtered_ideas)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content ideas for report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/{report_id}/ideas/{idea_id}/select")
async def select_content_idea(
    report_id: str,
    idea_id: str,
    selection_data: dict
):
    """
    Select a content idea for implementation
    
    Args:
        report_id: ID of the analysis report
        idea_id: ID of the content idea
        selection_data: Selection data and notes
        
    Returns:
        Selection confirmation
    """
    try:
        # Get content idea
        content_idea = database_service.get_content_idea_by_id(idea_id)
        
        if not content_idea:
            raise HTTPException(
                status_code=404,
                detail="Content idea not found"
            )
        
        # Update content idea with selection data
        updates = {
            "status": "selected",
            "selection_notes": selection_data.get("notes", ""),
            "priority": selection_data.get("priority", "medium"),
            "selected_at": datetime.utcnow().isoformat()
        }
        
        success = database_service.update_content_idea(idea_id, updates)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update content idea"
            )
        
        logger.info(f"Selected content idea {idea_id} for report {report_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Content idea selected successfully",
                "idea_id": idea_id,
                "status": "selected"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting content idea {idea_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/{report_id}/ideas/bulk-select")
async def bulk_select_content_ideas(
    report_id: str,
    selection_data: dict
):
    """
    Select multiple content ideas for implementation
    
    Args:
        report_id: ID of the analysis report
        selection_data: Selection data with idea IDs
        
    Returns:
        Bulk selection confirmation
    """
    try:
        idea_ids = selection_data.get("idea_ids", [])
        if not idea_ids:
            raise HTTPException(
                status_code=400,
                detail="No idea IDs provided"
            )
        
        selected_count = 0
        failed_ideas = []
        
        for idea_id in idea_ids:
            try:
                updates = {
                    "status": "selected",
                    "priority": selection_data.get("priority", "medium"),
                    "selected_at": datetime.utcnow().isoformat()
                }
                
                success = database_service.update_content_idea(idea_id, updates)
                if success:
                    selected_count += 1
                else:
                    failed_ideas.append(idea_id)
                    
            except Exception as e:
                logger.warning(f"Failed to select idea {idea_id}: {str(e)}")
                failed_ideas.append(idea_id)
        
        logger.info(f"Bulk selected {selected_count} content ideas for report {report_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Selected {selected_count} content ideas",
                "selected_count": selected_count,
                "failed_ideas": failed_ideas
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk selecting content ideas for report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/{report_id}/ideas/{idea_id}/indicators")
async def get_idea_indicators(
    report_id: str,
    idea_id: str
):
    """
    Get selection indicators for a content idea
    
    Args:
        report_id: ID of the analysis report
        idea_id: ID of the content idea
        
    Returns:
        Selection indicators for the content idea
    """
    try:
        # Get content idea
        content_idea = database_service.get_content_idea_by_id(idea_id)
        
        if not content_idea:
            raise HTTPException(
                status_code=404,
                detail="Content idea not found"
            )
        
        # Generate selection indicators
        indicators = _generate_selection_indicators(content_idea)
        
        return JSONResponse(
            status_code=200,
            content={
                "idea_id": idea_id,
                "indicators": indicators
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting indicators for idea {idea_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/{report_id}/ideas/compare")
async def compare_content_ideas(
    report_id: str,
    comparison_data: dict
):
    """
    Compare multiple content ideas
    
    Args:
        report_id: ID of the analysis report
        comparison_data: Data with idea IDs to compare
        
    Returns:
        Comparison results
    """
    try:
        idea_ids = comparison_data.get("idea_ids", [])
        if len(idea_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 ideas required for comparison"
            )
        
        # Get content ideas
        ideas = []
        for idea_id in idea_ids:
            idea = database_service.get_content_idea_by_id(idea_id)
            if idea:
                ideas.append(idea)
        
        if len(ideas) < 2:
            raise HTTPException(
                status_code=404,
                detail="Not enough valid ideas found for comparison"
            )
        
        # Generate comparison
        comparison = _generate_idea_comparison(ideas)
        
        return JSONResponse(
            status_code=200,
            content={
                "comparison": comparison,
                "ideas_compared": len(ideas)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing content ideas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/{report_id}/ideas/export")
async def export_content_ideas(
    report_id: str,
    export_data: dict
):
    """
    Export content ideas
    
    Args:
        report_id: ID of the analysis report
        export_data: Export configuration
        
    Returns:
        Exported content ideas
    """
    try:
        # Get content ideas
        content_ideas = database_service.get_content_ideas_by_report(report_id)
        
        if not content_ideas:
            raise HTTPException(
                status_code=404,
                detail="No content ideas found for this report"
            )
        
        # Apply filters if specified
        filtered_ideas = content_ideas
        if export_data.get("filters"):
            filtered_ideas = _apply_idea_filters(
                content_ideas, 
                export_data["filters"].get("content_type"),
                export_data["filters"].get("priority")
            )
        
        # Format for export
        export_format = export_data.get("format", "json")
        if export_format == "json":
            return JSONResponse(
                status_code=200,
                content={
                    "ideas": filtered_ideas,
                    "exported_at": datetime.utcnow().isoformat(),
                    "total_ideas": len(filtered_ideas)
                }
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="Unsupported export format"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting content ideas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

def _apply_idea_filters(
    ideas: List[Dict[str, Any]], 
    content_type: Optional[str], 
    priority: Optional[str]
) -> List[Dict[str, Any]]:
    """Apply filters to content ideas"""
    filtered_ideas = ideas
    
    if content_type:
        filtered_ideas = [idea for idea in filtered_ideas if idea.get("content_type") == content_type]
    
    if priority:
        filtered_ideas = [idea for idea in filtered_ideas if idea.get("priority") == priority]
    
    return filtered_ideas

def _sort_ideas(
    ideas: List[Dict[str, Any]], 
    sort_by: str, 
    sort_order: str
) -> List[Dict[str, Any]]:
    """Sort content ideas"""
    reverse = sort_order == "desc"
    
    if sort_by == "combined_score":
        ideas.sort(key=lambda x: x.get("combined_score", 0), reverse=reverse)
    elif sort_by == "seo_score":
        ideas.sort(key=lambda x: x.get("seo_optimization_score", 0), reverse=reverse)
    elif sort_by == "traffic_score":
        ideas.sort(key=lambda x: x.get("traffic_potential_score", 0), reverse=reverse)
    elif sort_by == "search_volume":
        ideas.sort(key=lambda x: x.get("total_search_volume", 0), reverse=reverse)
    elif sort_by == "created_at":
        ideas.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)
    
    return ideas

def _add_selection_indicators(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add selection indicators to content ideas"""
    for idea in ideas:
        idea["selection_indicators"] = _generate_selection_indicators(idea)
    
    return ideas

def _generate_selection_indicators(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate selection indicators for a content idea"""
    indicators = []
    
    # SEO Score indicator
    seo_score = idea.get("seo_optimization_score", 0)
    indicators.append({
        "type": "seo_score",
        "value": seo_score,
        "label": "SEO Score",
        "color": "green" if seo_score >= 80 else "orange" if seo_score >= 60 else "red",
        "description": f"SEO optimization score: {seo_score}/100"
    })
    
    # Traffic Potential indicator
    traffic_score = idea.get("traffic_potential_score", 0)
    indicators.append({
        "type": "traffic_potential",
        "value": traffic_score,
        "label": "Traffic Potential",
        "color": "blue" if traffic_score >= 80 else "orange" if traffic_score >= 60 else "red",
        "description": f"Traffic potential score: {traffic_score}/100"
    })
    
    # Difficulty indicator
    difficulty = idea.get("average_difficulty", 0)
    indicators.append({
        "type": "difficulty",
        "value": difficulty,
        "label": "Difficulty",
        "color": "green" if difficulty <= 30 else "orange" if difficulty <= 60 else "red",
        "description": f"Average keyword difficulty: {difficulty}/100"
    })
    
    # Search Volume indicator
    search_volume = idea.get("total_search_volume", 0)
    indicators.append({
        "type": "search_volume",
        "value": search_volume,
        "label": "Search Volume",
        "color": "purple" if search_volume >= 10000 else "blue" if search_volume >= 1000 else "gray",
        "description": f"Total monthly search volume: {search_volume:,}"
    })
    
    # CPC indicator
    cpc = idea.get("average_cpc", 0)
    indicators.append({
        "type": "cpc",
        "value": cpc,
        "label": "Average CPC",
        "color": "red" if cpc >= 3.0 else "orange" if cpc >= 1.0 else "green",
        "description": f"Average cost per click: ${cpc:.2f}"
    })
    
    return indicators

def _generate_idea_comparison(ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparison between content ideas"""
    comparison = {
        "metrics": {
            "seo_scores": [idea.get("seo_optimization_score", 0) for idea in ideas],
            "traffic_scores": [idea.get("traffic_potential_score", 0) for idea in ideas],
            "search_volumes": [idea.get("total_search_volume", 0) for idea in ideas],
            "difficulties": [idea.get("average_difficulty", 0) for idea in ideas],
            "cpcs": [idea.get("average_cpc", 0) for idea in ideas]
        },
        "rankings": {
            "best_seo": max(ideas, key=lambda x: x.get("seo_optimization_score", 0)),
            "best_traffic": max(ideas, key=lambda x: x.get("traffic_potential_score", 0)),
            "highest_volume": max(ideas, key=lambda x: x.get("total_search_volume", 0)),
            "easiest": min(ideas, key=lambda x: x.get("average_difficulty", 100)),
            "most_commercial": max(ideas, key=lambda x: x.get("average_cpc", 0))
        },
        "recommendations": _generate_comparison_recommendations(ideas)
    }
    
    return comparison

def _generate_comparison_recommendations(ideas: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on idea comparison"""
    recommendations = []
    
    # Find best overall idea
    best_idea = max(ideas, key=lambda x: x.get("combined_score", 0))
    recommendations.append(f"Best overall: {best_idea.get('title', 'Unknown')}")
    
    # Find quick wins
    quick_wins = [idea for idea in ideas if idea.get("average_difficulty", 100) <= 30]
    if quick_wins:
        recommendations.append(f"Quick wins available: {len(quick_wins)} ideas with low difficulty")
    
    # Find high-volume opportunities
    high_volume = [idea for idea in ideas if idea.get("total_search_volume", 0) >= 10000]
    if high_volume:
        recommendations.append(f"High-volume opportunities: {len(high_volume)} ideas with 10K+ search volume")
    
    # Find commercial opportunities
    commercial = [idea for idea in ideas if idea.get("average_cpc", 0) >= 2.0]
    if commercial:
        recommendations.append(f"Commercial opportunities: {len(commercial)} ideas with high CPC")
    
    return recommendations

def _organize_ideas_by_subtopic(ideas: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Organize ideas by sub-topic for better presentation"""
    subtopic_organization = {}
    
    for idea in ideas:
        # Extract sub-topic from primary keywords or content type
        subtopic = _determine_subtopic_from_idea(idea)
        
        if subtopic not in subtopic_organization:
            subtopic_organization[subtopic] = []
        
        subtopic_organization[subtopic].append(idea)
    
    # Sort ideas within each sub-topic by combined score
    for subtopic in subtopic_organization:
        subtopic_organization[subtopic].sort(
            key=lambda x: x.get("seo_optimization_score", 0) + x.get("traffic_potential_score", 0),
            reverse=True
        )
    
    return subtopic_organization

def _determine_subtopic_from_idea(idea: Dict[str, Any]) -> str:
    """Determine sub-topic from idea content"""
    primary_keywords = idea.get("primary_keywords", [])
    content_type = idea.get("content_type", "")
    
    if not primary_keywords:
        return "general"
    
    # Analyze primary keywords to determine sub-topic
    keyword_text = " ".join(primary_keywords).lower()
    
    if any(term in keyword_text for term in ["seo tool", "seo software", "keyword tool"]):
        return "seo_tools"
    elif any(term in keyword_text for term in ["content marketing", "content strategy", "blog"]):
        return "content_marketing"
    elif any(term in keyword_text for term in ["link building", "backlink", "link strategy"]):
        return "link_building"
    elif any(term in keyword_text for term in ["technical seo", "site speed", "mobile seo"]):
        return "technical_seo"
    elif any(term in keyword_text for term in ["local seo", "google my business", "local search"]):
        return "local_seo"
    elif any(term in keyword_text for term in ["ecommerce seo", "product seo", "shopify"]):
        return "ecommerce_seo"
    elif any(term in keyword_text for term in ["analytics", "tracking", "reporting"]):
        return "analytics_tracking"
    elif any(term in keyword_text for term in ["competitor", "competitor analysis"]):
        return "competitor_analysis"
    elif any(term in keyword_text for term in ["keyword research", "keyword analysis"]):
        return "keyword_research"
    else:
        return "general_seo"

