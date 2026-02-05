"""
ExportService for Google Docs, Notion, and WordPress integration
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.content_ideas import ContentIdeas
from ..models.software_solutions import SoftwareSolutions
from ..models.content_calendar import ContentCalendar
from ..models.export_templates import ExportTemplate

logger = structlog.get_logger()
settings = get_settings()

class ExportService:
    """Service for content export to various platforms"""
    
    def __init__(self):
        # API keys are now retrieved from database as needed
        self.google_docs_api_key = None
        self.notion_api_key = None
        self.wordpress_api_key = None
        self.wordpress_site_url = settings.wordpress_site_url
        
        # Export templates
        self.templates = {
            "google_docs": {
                "article": "Article Template",
                "software": "Software Solution Template",
                "calendar": "Calendar Template"
            },
            "notion": {
                "article": "Article Template",
                "software": "Software Solution Template",
                "calendar": "Calendar Template"
            },
            "wordpress": {
                "article": "Article Template",
                "software": "Software Solution Template",
                "calendar": "Calendar Template"
            }
        }
    
    async def export_to_google_docs(self, content_id: int, template_id: int, 
                                  custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export content to Google Docs"""
        try:
            # Get content and template
            db = next(get_db())
            content = db.get_ContentIdeas_by_id(ContentIdeas.id == content_id)
            template = db.get_ExportTemplate_by_id(ExportTemplate.id == template_id)
            
            if not content or not template:
                raise ValueError("Content or template not found")
            
            # Prepare export data
            export_data = self._prepare_export_data(content, template, custom_fields)
            
            # Create Google Docs document
            document_url = await self._create_google_docs_document(export_data, template)
            
            # Log export
            logger.info("Content exported to Google Docs", content_id=content_id, document_url=document_url)
            
            return {
                "success": True,
                "platform": "google_docs",
                "document_url": document_url,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export to Google Docs", content_id=content_id, error=str(e))
            raise
    
    async def export_to_notion(self, content_id: int, template_id: int, 
                             custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export content to Notion"""
        try:
            # Get content and template
            db = next(get_db())
            content = db.get_ContentIdeas_by_id(ContentIdeas.id == content_id)
            template = db.get_ExportTemplate_by_id(ExportTemplate.id == template_id)
            
            if not content or not template:
                raise ValueError("Content or template not found")
            
            # Prepare export data
            export_data = self._prepare_export_data(content, template, custom_fields)
            
            # Create Notion page
            page_url = await self._create_notion_page(export_data, template)
            
            # Log export
            logger.info("Content exported to Notion", content_id=content_id, page_url=page_url)
            
            return {
                "success": True,
                "platform": "notion",
                "page_url": page_url,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export to Notion", content_id=content_id, error=str(e))
            raise
    
    async def export_to_wordpress(self, content_id: int, template_id: int, 
                                custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export content to WordPress"""
        try:
            # Get content and template
            db = next(get_db())
            content = db.get_ContentIdeas_by_id(ContentIdeas.id == content_id)
            template = db.get_ExportTemplate_by_id(ExportTemplate.id == template_id)
            
            if not content or not template:
                raise ValueError("Content or template not found")
            
            # Prepare export data
            export_data = self._prepare_export_data(content, template, custom_fields)
            
            # Create WordPress post
            post_url = await self._create_wordpress_post(export_data, template)
            
            # Log export
            logger.info("Content exported to WordPress", content_id=content_id, post_url=post_url)
            
            return {
                "success": True,
                "platform": "wordpress",
                "post_url": post_url,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export to WordPress", content_id=content_id, error=str(e))
            raise
    
    async def export_software_solution(self, software_solution_id: int, platform: str, 
                                     template_id: int, custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export software solution to platform"""
        try:
            # Get software solution and template
            db = next(get_db())
            software_solution = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == software_solution_id)
            template = db.get_ExportTemplate_by_id(ExportTemplate.id == template_id)
            
            if not software_solution or not template:
                raise ValueError("Software solution or template not found")
            
            # Prepare export data
            export_data = self._prepare_software_export_data(software_solution, template, custom_fields)
            
            # Export to platform
            if platform == "google_docs":
                result = await self._create_google_docs_document(export_data, template)
            elif platform == "notion":
                result = await self._create_notion_page(export_data, template)
            elif platform == "wordpress":
                result = await self._create_wordpress_post(export_data, template)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            # Log export
            logger.info("Software solution exported", software_solution_id=software_solution_id, platform=platform)
            
            return {
                "success": True,
                "platform": platform,
                "export_url": result,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export software solution", software_solution_id=software_solution_id, error=str(e))
            raise
    
    async def export_calendar_entries(self, user_id: int, start_date: datetime, end_date: datetime,
                                    platform: str, template_id: int) -> Dict[str, Any]:
        """Export calendar entries to platform"""
        try:
            # Get calendar entries
            db = next(get_db())
            calendar_entries = db.query(ContentCalendar).filter(
                ContentCalendar.user_id == user_id,
                ContentCalendar.scheduled_date >= start_date,
                ContentCalendar.scheduled_date <= end_date
            ).all()
            
            if not calendar_entries:
                raise ValueError("No calendar entries found for date range")
            
            # Get template
            template = db.get_ExportTemplate_by_id(ExportTemplate.id == template_id)
            if not template:
                raise ValueError("Template not found")
            
            # Prepare export data
            export_data = self._prepare_calendar_export_data(calendar_entries, template)
            
            # Export to platform
            if platform == "google_docs":
                result = await self._create_google_docs_document(export_data, template)
            elif platform == "notion":
                result = await self._create_notion_page(export_data, template)
            elif platform == "wordpress":
                result = await self._create_wordpress_post(export_data, template)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            # Log export
            logger.info("Calendar entries exported", user_id=user_id, platform=platform, entries_count=len(calendar_entries))
            
            return {
                "success": True,
                "platform": platform,
                "export_url": result,
                "entries_count": len(calendar_entries),
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to export calendar entries", user_id=user_id, error=str(e))
            raise
    
    def _prepare_export_data(self, content: ContentIdeas, template: ExportTemplate, 
                           custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare content data for export"""
        # Get article angles
        article_angles = content.article_angles or []
        
        # Get content outlines
        content_outlines = content.content_outlines or {}
        
        # Get SEO recommendations
        seo_recommendations = content.seo_recommendations or {}
        
        # Prepare export data
        export_data = {
            "title": content.title or "Generated Content",
            "article_angles": article_angles,
            "content_outlines": content_outlines,
            "seo_recommendations": seo_recommendations,
            "opportunity_score": content.opportunity_score,
            "created_at": content.created_at.isoformat(),
            "custom_fields": custom_fields or {}
        }
        
        # Add template-specific data
        if template.template_type == "article":
            export_data["content_type"] = "article"
        elif template.template_type == "software":
            export_data["content_type"] = "software"
        elif template.template_type == "calendar":
            export_data["content_type"] = "calendar"
        
        return export_data
    
    def _prepare_software_export_data(self, software_solution: SoftwareSolutions, template: ExportTemplate,
                                    custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare software solution data for export"""
        # Get software solutions
        software_solutions = software_solution.software_solutions or []
        
        # Prepare export data
        export_data = {
            "title": f"Software Solutions - {software_solution.keyword_data.topic if software_solution.keyword_data else 'Generated'}",
            "software_solutions": software_solutions,
            "created_at": software_solution.created_at.isoformat(),
            "custom_fields": custom_fields or {}
        }
        
        return export_data
    
    def _prepare_calendar_export_data(self, calendar_entries: List[ContentCalendar], template: ExportTemplate) -> Dict[str, Any]:
        """Prepare calendar entries data for export"""
        # Prepare export data
        export_data = {
            "title": "Content Calendar",
            "calendar_entries": [
                {
                    "id": entry.id,
                    "title": entry.title,
                    "scheduled_date": entry.scheduled_date.isoformat(),
                    "status": entry.status,
                    "notes": entry.notes,
                    "content_type": entry.content_type
                }
                for entry in calendar_entries
            ],
            "exported_at": datetime.utcnow().isoformat()
        }
        
        return export_data
    
    async def _create_google_docs_document(self, export_data: Dict[str, Any], template: ExportTemplate) -> str:
        """Create Google Docs document"""
        try:
            # Prepare document content
            content = self._format_content_for_google_docs(export_data, template)
            
            # Create document via Google Docs API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.google_docs_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Create document
                create_data = {
                    "title": export_data.get("title", "Exported Content"),
                    "body": {
                        "content": content
                    }
                }
                
                async with session.post(
                    "https://docs.googleapis.com/v1/documents",
                    headers=headers,
                    json=create_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("documentId", "")
                    else:
                        raise Exception(f"Google Docs API error: {response.status}")
            
        except Exception as e:
            logger.error("Failed to create Google Docs document", error=str(e))
            raise
    
    async def _create_notion_page(self, export_data: Dict[str, Any], template: ExportTemplate) -> str:
        """Create Notion page"""
        try:
            # Prepare page content
            content = self._format_content_for_notion(export_data, template)
            
            # Create page via Notion API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.notion_api_key}",
                    "Content-Type": "application/json",
                    "Notion-Version": "2022-06-28"
                }
                
                # Create page
                create_data = {
                    "parent": {"database_id": template.notion_database_id},
                    "properties": {
                        "title": {
                            "title": [
                                {
                                    "text": {
                                        "content": export_data.get("title", "Exported Content")
                                    }
                                }
                            ]
                        }
                    },
                    "children": content
                }
                
                async with session.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json=create_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("url", "")
                    else:
                        raise Exception(f"Notion API error: {response.status}")
            
        except Exception as e:
            logger.error("Failed to create Notion page", error=str(e))
            raise
    
    async def _create_wordpress_post(self, export_data: Dict[str, Any], template: ExportTemplate) -> str:
        """Create WordPress post"""
        try:
            # Prepare post content
            content = self._format_content_for_wordpress(export_data, template)
            
            # Create post via WordPress REST API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.wordpress_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Create post
                create_data = {
                    "title": export_data.get("title", "Exported Content"),
                    "content": content,
                    "status": "draft",
                    "format": "standard"
                }
                
                async with session.post(
                    f"{self.wordpress_site_url}/wp-json/wp/v2/posts",
                    headers=headers,
                    json=create_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return result.get("link", "")
                    else:
                        raise Exception(f"WordPress API error: {response.status}")
            
        except Exception as e:
            logger.error("Failed to create WordPress post", error=str(e))
            raise
    
    def _format_content_for_google_docs(self, export_data: Dict[str, Any], template: ExportTemplate) -> List[Dict[str, Any]]:
        """Format content for Google Docs"""
        content = []
        
        # Add title
        content.append({
            "paragraph": {
                "elements": [
                    {
                        "textRun": {
                            "content": export_data.get("title", "Exported Content"),
                            "textStyle": {
                                "bold": True,
                                "fontSize": {"magnitude": 18, "unit": "PT"}
                            }
                        }
                    }
                ]
            }
        })
        
        # Add content based on type
        if export_data.get("content_type") == "article":
            content.extend(self._format_article_for_google_docs(export_data))
        elif export_data.get("content_type") == "software":
            content.extend(self._format_software_for_google_docs(export_data))
        elif export_data.get("content_type") == "calendar":
            content.extend(self._format_calendar_for_google_docs(export_data))
        
        return content
    
    def _format_content_for_notion(self, export_data: Dict[str, Any], template: ExportTemplate) -> List[Dict[str, Any]]:
        """Format content for Notion"""
        content = []
        
        # Add title
        content.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": export_data.get("title", "Exported Content")
                        }
                    }
                ]
            }
        })
        
        # Add content based on type
        if export_data.get("content_type") == "article":
            content.extend(self._format_article_for_notion(export_data))
        elif export_data.get("content_type") == "software":
            content.extend(self._format_software_for_notion(export_data))
        elif export_data.get("content_type") == "calendar":
            content.extend(self._format_calendar_for_notion(export_data))
        
        return content
    
    def _format_content_for_wordpress(self, export_data: Dict[str, Any], template: ExportTemplate) -> str:
        """Format content for WordPress"""
        content = f"<h1>{export_data.get('title', 'Exported Content')}</h1>\n\n"
        
        # Add content based on type
        if export_data.get("content_type") == "article":
            content += self._format_article_for_wordpress(export_data)
        elif export_data.get("content_type") == "software":
            content += self._format_software_for_wordpress(export_data)
        elif export_data.get("content_type") == "calendar":
            content += self._format_calendar_for_wordpress(export_data)
        
        return content
    
    def _format_article_for_google_docs(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format article content for Google Docs"""
        content = []
        
        # Add article angles
        article_angles = export_data.get("article_angles", [])
        if article_angles:
            content.append({
                "paragraph": {
                    "elements": [
                        {
                            "textRun": {
                                "content": "Article Angles:",
                                "textStyle": {"bold": True}
                            }
                        }
                    ]
                }
            })
            
            for angle in article_angles:
                content.append({
                    "paragraph": {
                        "elements": [
                            {
                                "textRun": {
                                    "content": f"• {angle.get('title', 'Untitled')}",
                                    "textStyle": {"italic": True}
                                }
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_article_for_notion(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format article content for Notion"""
        content = []
        
        # Add article angles
        article_angles = export_data.get("article_angles", [])
        if article_angles:
            content.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "Article Angles"}
                        }
                    ]
                }
            })
            
            for angle in article_angles:
                content.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": angle.get("title", "Untitled")}
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_article_for_wordpress(self, export_data: Dict[str, Any]) -> str:
        """Format article content for WordPress"""
        content = ""
        
        # Add article angles
        article_angles = export_data.get("article_angles", [])
        if article_angles:
            content += "<h2>Article Angles</h2>\n<ul>\n"
            for angle in article_angles:
                content += f"<li>{angle.get('title', 'Untitled')}</li>\n"
            content += "</ul>\n\n"
        
        return content
    
    def _format_software_for_google_docs(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format software content for Google Docs"""
        content = []
        
        # Add software solutions
        software_solutions = export_data.get("software_solutions", [])
        if software_solutions:
            content.append({
                "paragraph": {
                    "elements": [
                        {
                            "textRun": {
                                "content": "Software Solutions:",
                                "textStyle": {"bold": True}
                            }
                        }
                    ]
                }
            })
            
            for solution in software_solutions:
                content.append({
                    "paragraph": {
                        "elements": [
                            {
                                "textRun": {
                                    "content": f"• {solution.get('name', 'Untitled')}",
                                    "textStyle": {"italic": True}
                                }
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_software_for_notion(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format software content for Notion"""
        content = []
        
        # Add software solutions
        software_solutions = export_data.get("software_solutions", [])
        if software_solutions:
            content.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "Software Solutions"}
                        }
                    ]
                }
            })
            
            for solution in software_solutions:
                content.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": solution.get("name", "Untitled")}
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_software_for_wordpress(self, export_data: Dict[str, Any]) -> str:
        """Format software content for WordPress"""
        content = ""
        
        # Add software solutions
        software_solutions = export_data.get("software_solutions", [])
        if software_solutions:
            content += "<h2>Software Solutions</h2>\n<ul>\n"
            for solution in software_solutions:
                content += f"<li>{solution.get('name', 'Untitled')}</li>\n"
            content += "</ul>\n\n"
        
        return content
    
    def _format_calendar_for_google_docs(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format calendar content for Google Docs"""
        content = []
        
        # Add calendar entries
        calendar_entries = export_data.get("calendar_entries", [])
        if calendar_entries:
            content.append({
                "paragraph": {
                    "elements": [
                        {
                            "textRun": {
                                "content": "Calendar Entries:",
                                "textStyle": {"bold": True}
                            }
                        }
                    ]
                }
            })
            
            for entry in calendar_entries:
                content.append({
                    "paragraph": {
                        "elements": [
                            {
                                "textRun": {
                                    "content": f"• {entry.get('title', 'Untitled')} - {entry.get('scheduled_date', 'No date')}",
                                    "textStyle": {"italic": True}
                                }
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_calendar_for_notion(self, export_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format calendar content for Notion"""
        content = []
        
        # Add calendar entries
        calendar_entries = export_data.get("calendar_entries", [])
        if calendar_entries:
            content.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "Calendar Entries"}
                        }
                    ]
                }
            })
            
            for entry in calendar_entries:
                content.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": f"{entry.get('title', 'Untitled')} - {entry.get('scheduled_date', 'No date')}"}
                            }
                        ]
                    }
                })
        
        return content
    
    def _format_calendar_for_wordpress(self, export_data: Dict[str, Any]) -> str:
        """Format calendar content for WordPress"""
        content = ""
        
        # Add calendar entries
        calendar_entries = export_data.get("calendar_entries", [])
        if calendar_entries:
            content += "<h2>Calendar Entries</h2>\n<ul>\n"
            for entry in calendar_entries:
                content += f"<li>{entry.get('title', 'Untitled')} - {entry.get('scheduled_date', 'No date')}</li>\n"
            content += "</ul>\n\n"
        
        return content
