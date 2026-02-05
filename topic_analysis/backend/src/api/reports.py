"""
Reports API Endpoints

Handles report generation, retrieval, and export functionality.
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import logging
import json
import csv
from datetime import datetime
from io import StringIO

from ..models.analysis_report import KeywordAnalysisReport
from ..services.report_generator import ReportGenerator
from ..services.database import DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# Initialize services
report_generator = ReportGenerator()
database = DatabaseService()

@router.get("/{report_id}")
async def get_report(report_id: str) -> Dict[str, Any]:
    """
    Get analysis report by ID
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Report data
    """
    try:
        # Get analysis report
        analysis_report = await database.get_analysis_report(report_id)
        if not analysis_report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get associated data
        keywords = await database.get_keywords_by_analysis_id(report_id)
        content_opportunities = await database.get_content_opportunities_by_analysis_id(report_id)
        seo_content_ideas = await database.get_seo_content_ideas_by_analysis_id(report_id)
        
        # Generate report summary
        report_summary = report_generator.generate_summary(
            analysis_report, 
            keywords, 
            content_opportunities, 
            seo_content_ideas
        )
        
        return {
            "report_id": analysis_report.id,
            "file_id": analysis_report.file_id,
            "user_id": analysis_report.user_id,
            "status": analysis_report.status,
            "summary": report_summary,
            "keywords_count": analysis_report.keywords_analyzed,
            "content_opportunities_count": analysis_report.content_opportunities_count,
            "seo_content_ideas_count": analysis_report.seo_content_ideas_count,
            "created_at": analysis_report.created_at.isoformat(),
            "completed_at": analysis_report.completed_at.isoformat() if analysis_report.completed_at else None,
            "error_message": analysis_report.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report retrieval failed: {str(e)}")

@router.get("/{report_id}/export")
async def export_report(
    report_id: str, 
    format: str = "json"
) -> Response:
    """
    Export report in specified format
    
    Args:
        report_id: Unique report identifier
        format: Export format (json, csv, pdf)
        
    Returns:
        Exported report data
    """
    try:
        # Get analysis report
        analysis_report = await database.get_analysis_report(report_id)
        if not analysis_report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if analysis_report.status != "completed":
            raise HTTPException(status_code=400, detail="Report not completed")
        
        # Get associated data
        keywords = await database.get_keywords_by_analysis_id(report_id)
        content_opportunities = await database.get_content_opportunities_by_analysis_id(report_id)
        seo_content_ideas = await database.get_seo_content_ideas_by_analysis_id(report_id)
        
        if format.lower() == "json":
            return await _export_json(analysis_report, keywords, content_opportunities, seo_content_ideas)
        elif format.lower() == "csv":
            return await _export_csv(analysis_report, keywords, content_opportunities, seo_content_ideas)
        elif format.lower() == "pdf":
            return await _export_pdf(analysis_report, keywords, content_opportunities, seo_content_ideas)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

async def _export_json(
    analysis_report: KeywordAnalysisReport,
    keywords: List,
    content_opportunities: List,
    seo_content_ideas: List
) -> JSONResponse:
    """Export report as JSON"""
    export_data = {
        "report_id": analysis_report.id,
        "file_id": analysis_report.file_id,
        "exported_at": datetime.utcnow().isoformat(),
        "summary": {
            "keywords_count": len(keywords),
            "content_opportunities_count": len(content_opportunities),
            "seo_content_ideas_count": len(seo_content_ideas)
        },
        "keywords": [
            {
                "keyword": k.keyword,
                "volume": k.volume,
                "difficulty": k.difficulty,
                "cpc": k.cpc,
                "intents": k.intents,
                "opportunity_score": k.opportunity_score
            }
            for k in keywords
        ],
        "content_opportunities": [
            {
                "keyword": co.keyword,
                "opportunity_score": co.opportunity_score,
                "content_suggestions": co.content_suggestions,
                "priority": co.priority
            }
            for co in content_opportunities
        ],
        "seo_content_ideas": [
            {
                "title": idea.title,
                "content_type": idea.content_type,
                "primary_keywords": idea.primary_keywords,
                "secondary_keywords": idea.secondary_keywords,
                "seo_optimization_score": idea.seo_optimization_score,
                "traffic_potential_score": idea.traffic_potential_score,
                "total_search_volume": idea.total_search_volume,
                "average_difficulty": idea.average_difficulty,
                "average_cpc": idea.average_cpc,
                "optimization_tips": idea.optimization_tips,
                "content_outline": idea.content_outline
            }
            for idea in seo_content_ideas
        ]
    }
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=report_{analysis_report.id}.json"
        }
    )

async def _export_csv(
    analysis_report: KeywordAnalysisReport,
    keywords: List,
    content_opportunities: List,
    seo_content_ideas: List
) -> Response:
    """Export report as CSV"""
    output = StringIO()
    
    # Write keywords CSV
    if keywords:
        writer = csv.writer(output)
        writer.writerow(["Keyword", "Volume", "Difficulty", "CPC", "Intents", "Opportunity Score"])
        for k in keywords:
            writer.writerow([
                k.keyword,
                k.volume,
                k.difficulty,
                k.cpc,
                ",".join(k.intents),
                k.opportunity_score
            ])
    
    csv_content = output.getvalue()
    output.close()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=keywords_{analysis_report.id}.csv"
        }
    )

async def _export_pdf(
    analysis_report: KeywordAnalysisReport,
    keywords: List,
    content_opportunities: List,
    seo_content_ideas: List
) -> Response:
    """Export report as PDF (placeholder - would need PDF generation library)"""
    # This is a placeholder implementation
    # In a real implementation, you would use a library like reportlab or weasyprint
    # to generate a proper PDF report
    
    pdf_content = f"""
    Keyword Analysis Report
    Report ID: {analysis_report.id}
    Generated: {datetime.utcnow().isoformat()}
    
    Summary:
    - Keywords: {len(keywords)}
    - Content Opportunities: {len(content_opportunities)}
    - SEO Content Ideas: {len(seo_content_ideas)}
    """
    
    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=report_{analysis_report.id}.pdf"
        }
    )

@router.get("/")
async def list_reports(
    user_id: str = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List reports for a user
    
    Args:
        user_id: User ID to filter reports
        limit: Maximum number of reports to return
        offset: Number of reports to skip
        
    Returns:
        List of reports
    """
    try:
        reports = await database.list_analysis_reports(user_id, limit, offset)
        total_count = await database.count_analysis_reports(user_id)
        
        return {
            "reports": [
                {
                    "report_id": r.id,
                    "file_id": r.file_id,
                    "status": r.status,
                    "keywords_count": r.keywords_analyzed,
                    "content_opportunities_count": r.content_opportunities_count,
                    "seo_content_ideas_count": r.seo_content_ideas_count,
                    "created_at": r.created_at.isoformat(),
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None
                }
                for r in reports
            ],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report listing failed: {str(e)}")

@router.delete("/{report_id}")
async def delete_report(report_id: str) -> Dict[str, Any]:
    """
    Delete a report and its associated data
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Get analysis report
        analysis_report = await database.get_analysis_report(report_id)
        if not analysis_report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Delete associated data
        await database.delete_analysis_data(report_id)
        
        # Delete report
        await database.delete_analysis_report(report_id)
        
        logger.info(f"Report deleted: {report_id}")
        
        return {
            "report_id": report_id,
            "status": "deleted",
            "message": "Report and associated data deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report deletion failed: {str(e)}")

