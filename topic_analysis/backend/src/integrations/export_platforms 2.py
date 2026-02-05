"""
Export Platforms Integration
Integrates with Google Docs, Notion, WordPress, and other content platforms
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class ExportPlatform:
    """Base class for export platform integrations"""
    
    def __init__(self, platform_name: str, api_key: str, base_url: str):
        self.platform_name = platform_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = 60.0
    
    async def export_content(
        self,
        content: Dict[str, Any],
        title: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to the platform"""
        raise NotImplementedError("Subclasses must implement export_content")
    
    async def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get status of an export"""
        raise NotImplementedError("Subclasses must implement get_export_status")

class GoogleDocsAPI(ExportPlatform):
    """Google Docs API integration"""
    
    def __init__(self):
        super().__init__(
            "Google Docs",
            settings.google_docs_api_key,
            "https://docs.googleapis.com/v1"
        )
    
    async def export_content(
        self,
        content: Dict[str, Any],
        title: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to Google Docs"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/documents"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Create document structure
                document_body = {
                    "title": title,
                    "body": {
                        "content": self._format_content_for_docs(content)
                    }
                }
                
                response = await client.post(url, json=document_body, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_google_docs_response(data, title)
                
        except Exception as e:
            logger.error(f"Google Docs API error: {e}")
            return {"error": str(e), "platform": self.platform_name, "title": title}
    
    async def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get Google Docs export status"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/documents/{export_id}"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return {
                    "export_id": export_id,
                    "status": "completed",
                    "url": data.get("webViewLink", ""),
                    "platform": self.platform_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Google Docs status check error: {e}")
            return {"error": str(e), "export_id": export_id}
    
    def _format_content_for_docs(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format content for Google Docs structure"""
        elements = []
        
        # Add title
        if "title" in content:
            elements.append({
                "insertText": {
                    "text": content["title"] + "\n",
                    "location": {"index": 1}
                }
            })
            elements.append({
                "updateTextStyle": {
                    "range": {"startIndex": 1, "endIndex": len(content["title"]) + 1},
                    "textStyle": {"bold": True, "fontSize": {"magnitude": 18, "unit": "PT"}}
                }
            })
        
        # Add content sections
        if "sections" in content:
            for section in content["sections"]:
                # Section title
                elements.append({
                    "insertText": {
                        "text": f"\n{section.get('title', '')}\n",
                        "location": {"index": 1}
                    }
                })
                elements.append({
                    "updateTextStyle": {
                        "range": {"startIndex": 1, "endIndex": len(section.get('title', '')) + 1},
                        "textStyle": {"bold": True, "fontSize": {"magnitude": 14, "unit": "PT"}}
                    }
                })
                
                # Section content
                if "content" in section:
                    elements.append({
                        "insertText": {
                            "text": section["content"] + "\n",
                            "location": {"index": 1}
                        }
                    })
        
        return elements
    
    def _process_google_docs_response(self, data: Dict[str, Any], title: str) -> Dict[str, Any]:
        """Process Google Docs API response"""
        return {
            "export_id": data.get("documentId", ""),
            "title": title,
            "url": data.get("webViewLink", ""),
            "platform": self.platform_name,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }

class NotionAPI(ExportPlatform):
    """Notion API integration"""
    
    def __init__(self):
        super().__init__(
            "Notion",
            settings.notion_api_key,
            "https://api.notion.com/v1"
        )
        self.notion_version = "2022-06-28"
    
    async def export_content(
        self,
        content: Dict[str, Any],
        title: str,
        database_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to Notion"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/pages"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Notion-Version": self.notion_version
                }
                
                # Create page structure
                page_data = {
                    "parent": {"database_id": database_id} if database_id else {"type": "page_id", "page_id": "root"},
                    "properties": {
                        "title": {
                            "title": [{"text": {"content": title}}]
                        }
                    },
                    "children": self._format_content_for_notion(content)
                }
                
                response = await client.post(url, json=page_data, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_notion_response(data, title)
                
        except Exception as e:
            logger.error(f"Notion API error: {e}")
            return {"error": str(e), "platform": self.platform_name, "title": title}
    
    async def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get Notion export status"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/pages/{export_id}"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Notion-Version": self.notion_version
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return {
                    "export_id": export_id,
                    "status": "completed",
                    "url": data.get("url", ""),
                    "platform": self.platform_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Notion status check error: {e}")
            return {"error": str(e), "export_id": export_id}
    
    def _format_content_for_notion(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format content for Notion structure"""
        blocks = []
        
        # Add content sections
        if "sections" in content:
            for section in content["sections"]:
                # Section title
                if "title" in section:
                    blocks.append({
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{"type": "text", "text": {"content": section["title"]}}]
                        }
                    })
                
                # Section content
                if "content" in section:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": section["content"]}}]
                        }
                    })
        
        return blocks
    
    def _process_notion_response(self, data: Dict[str, Any], title: str) -> Dict[str, Any]:
        """Process Notion API response"""
        return {
            "export_id": data.get("id", ""),
            "title": title,
            "url": data.get("url", ""),
            "platform": self.platform_name,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }

class WordPressAPI(ExportPlatform):
    """WordPress API integration"""
    
    def __init__(self):
        super().__init__(
            "WordPress",
            settings.wordpress_api_key,
            settings.wordpress_api_url
        )
        self.username = settings.wordpress_username
        self.password = settings.wordpress_password
    
    async def export_content(
        self,
        content: Dict[str, Any],
        title: str,
        status: str = "draft",
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to WordPress"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/wp-json/wp/v2/posts"
                
                # WordPress uses Basic Auth
                auth = httpx.BasicAuth(self.username, self.password)
                
                # Format content for WordPress
                post_content = self._format_content_for_wordpress(content)
                
                post_data = {
                    "title": title,
                    "content": post_content,
                    "status": status,
                    "format": "standard"
                }
                
                response = await client.post(url, json=post_data, auth=auth)
                response.raise_for_status()
                
                data = response.json()
                return self._process_wordpress_response(data, title)
                
        except Exception as e:
            logger.error(f"WordPress API error: {e}")
            return {"error": str(e), "platform": self.platform_name, "title": title}
    
    async def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get WordPress export status"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/wp-json/wp/v2/posts/{export_id}"
                
                auth = httpx.BasicAuth(self.username, self.password)
                
                response = await client.get(url, auth=auth)
                response.raise_for_status()
                
                data = response.json()
                return {
                    "export_id": str(export_id),
                    "status": data.get("status", "unknown"),
                    "url": data.get("link", ""),
                    "platform": self.platform_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"WordPress status check error: {e}")
            return {"error": str(e), "export_id": export_id}
    
    def _format_content_for_wordpress(self, content: Dict[str, Any]) -> str:
        """Format content for WordPress"""
        html_content = ""
        
        # Add content sections
        if "sections" in content:
            for section in content["sections"]:
                # Section title
                if "title" in section:
                    html_content += f"<h2>{section['title']}</h2>\n"
                
                # Section content
                if "content" in section:
                    html_content += f"<p>{section['content']}</p>\n"
        
        return html_content
    
    def _process_wordpress_response(self, data: Dict[str, Any], title: str) -> Dict[str, Any]:
        """Process WordPress API response"""
        return {
            "export_id": str(data.get("id", "")),
            "title": title,
            "url": data.get("link", ""),
            "platform": self.platform_name,
            "status": data.get("status", "draft"),
            "created_at": datetime.utcnow().isoformat()
        }

class ExportPlatformsManager:
    """Manages all export platform integrations"""
    
    def __init__(self):
        self.platforms = {
            "google_docs": GoogleDocsAPI(),
            "notion": NotionAPI(),
            "wordpress": WordPressAPI()
        }
    
    async def export_to_platform(
        self,
        platform: str,
        content: Dict[str, Any],
        title: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to a specific platform"""
        if platform not in self.platforms:
            return {"error": f"Unknown platform: {platform}"}
        
        try:
            platform_api = self.platforms[platform]
            return await platform_api.export_content(content, title, **kwargs)
        except Exception as e:
            logger.error(f"Export to {platform} error: {e}")
            return {"error": str(e), "platform": platform}
    
    async def export_to_multiple_platforms(
        self,
        platforms: List[str],
        content: Dict[str, Any],
        title: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Export content to multiple platforms"""
        results = {}
        
        # Run exports in parallel
        tasks = []
        for platform in platforms:
            if platform in self.platforms:
                task = self.export_to_platform(platform, content, title, **kwargs)
                tasks.append((platform, task))
        
        export_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for i, (platform, _) in enumerate(tasks):
            result = export_results[i]
            if isinstance(result, Exception):
                results[platform] = {"error": str(result)}
            else:
                results[platform] = result
        
        return {
            "content": content,
            "title": title,
            "exports": results,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def get_export_status(
        self,
        platform: str,
        export_id: str
    ) -> Dict[str, Any]:
        """Get export status for a specific platform"""
        if platform not in self.platforms:
            return {"error": f"Unknown platform: {platform}"}
        
        try:
            platform_api = self.platforms[platform]
            return await platform_api.get_export_status(export_id)
        except Exception as e:
            logger.error(f"Get export status for {platform} error: {e}")
            return {"error": str(e), "platform": platform}
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available export platforms"""
        return list(self.platforms.keys())
    
    async def test_platforms(self) -> Dict[str, Dict[str, Any]]:
        """Test all export platforms"""
        test_content = {
            "title": "Test Export",
            "sections": [
                {
                    "title": "Test Section",
                    "content": "This is a test export to verify platform connectivity."
                }
            ]
        }
        
        results = {}
        
        for platform_name, platform_api in self.platforms.items():
            try:
                result = await platform_api.export_content(test_content, "Test Export")
                results[platform_name] = {
                    "status": "success" if "error" not in result else "error",
                    "result": result
                }
            except Exception as e:
                results[platform_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results

# Global instance
export_platforms_manager = ExportPlatformsManager()

# Convenience functions
async def export_content(
    platform: str,
    content: Dict[str, Any],
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """Export content to a specific platform"""
    return await export_platforms_manager.export_to_platform(platform, content, title, **kwargs)

async def export_to_multiple_platforms(
    platforms: List[str],
    content: Dict[str, Any],
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """Export content to multiple platforms"""
    return await export_platforms_manager.export_to_multiple_platforms(platforms, content, title, **kwargs)

async def get_export_status(platform: str, export_id: str) -> Dict[str, Any]:
    """Get export status for a specific platform"""
    return await export_platforms_manager.get_export_status(platform, export_id)

def get_available_platforms() -> List[str]:
    """Get list of available export platforms"""
    return export_platforms_manager.get_available_platforms()

async def test_all_platforms() -> Dict[str, Dict[str, Any]]:
    """Test all export platforms"""
    return await export_platforms_manager.test_platforms()
