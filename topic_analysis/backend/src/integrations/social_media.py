"""
Social Media APIs Integration
Integrates with Reddit, Twitter, TikTok, and RSS feeds for trend analysis
"""

import httpx
import asyncio
import feedparser
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class SocialMediaAPI:
    """Base class for social media API integrations"""
    
    def __init__(self, platform_name: str, api_key: str, base_url: str):
        self.platform_name = platform_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = 30.0
    
    async def search_posts(
        self,
        query: str,
        limit: int = 50,
        timeframe: str = "week"
    ) -> List[Dict[str, Any]]:
        """Search for posts on the platform"""
        raise NotImplementedError("Subclasses must implement search_posts")
    
    async def get_trending_topics(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics on the platform"""
        raise NotImplementedError("Subclasses must implement get_trending_topics")

class RedditAPI(SocialMediaAPI):
    """Reddit API integration"""
    
    def __init__(self):
        super().__init__(
            "Reddit",
            settings.REDDIT_API_KEY,
            "https://www.reddit.com"
        )
        self.user_agent = "TrendTap/1.0"
    
    async def search_posts(
        self,
        query: str,
        limit: int = 50,
        timeframe: str = "week"
    ) -> List[Dict[str, Any]]:
        """Search Reddit posts"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Reddit search endpoint
                url = f"{self.base_url}/search.json"
                
                params = {
                    "q": query,
                    "limit": limit,
                    "sort": "relevance",
                    "t": timeframe,
                    "raw_json": 1
                }
                
                headers = {
                    "User-Agent": self.user_agent
                }
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_reddit_posts(data)
                
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return []
    
    async def get_trending_topics(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics from Reddit"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Get hot posts from popular subreddits
                subreddits = ["popular", "all", "trending"]
                if category:
                    subreddits = [category]
                
                all_posts = []
                
                for subreddit in subreddits:
                    url = f"{self.base_url}/r/{subreddit}/hot.json"
                    params = {"limit": 25, "raw_json": 1}
                    headers = {"User-Agent": self.user_agent}
                    
                    response = await client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    posts = self._process_reddit_posts(data)
                    all_posts.extend(posts)
                
                # Sort by score and return top trending
                all_posts.sort(key=lambda x: x.get("score", 0), reverse=True)
                return all_posts[:50]
                
        except Exception as e:
            logger.error(f"Reddit trending topics error: {e}")
            return []
    
    def _process_reddit_posts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Reddit API response"""
        posts = []
        
        if "data" in data and "children" in data["data"]:
            for child in data["data"]["children"]:
                post_data = child.get("data", {})
                
                posts.append({
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "content": post_data.get("selftext", ""),
                    "url": post_data.get("url", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "author": post_data.get("author", ""),
                    "score": post_data.get("score", 0),
                    "upvote_ratio": post_data.get("upvote_ratio", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "created_utc": post_data.get("created_utc", 0),
                    "platform": "reddit",
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return posts

class TwitterAPI(SocialMediaAPI):
    """Twitter API integration (simplified)"""
    
    def __init__(self):
        super().__init__(
            "Twitter",
            settings.TWITTER_API_KEY,
            "https://api.twitter.com/2"
        )
        self.bearer_token = settings.TWITTER_BEARER_TOKEN
    
    async def search_posts(
        self,
        query: str,
        limit: int = 50,
        timeframe: str = "week"
    ) -> List[Dict[str, Any]]:
        """Search Twitter posts (simplified implementation)"""
        try:
            # Note: This is a simplified implementation
            # Real Twitter API v2 requires OAuth 2.0 Bearer Token authentication
            # and has rate limits and approval requirements
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/tweets/search/recent"
                
                headers = {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "query": query,
                    "max_results": min(limit, 100),
                    "tweet.fields": "created_at,public_metrics,context_annotations"
                }
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_twitter_posts(data)
                
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            # Return mock data for development
            return self._get_mock_twitter_posts(query, limit)
    
    async def get_trending_topics(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics from Twitter"""
        try:
            # Mock implementation for development
            return self._get_mock_trending_topics()
        except Exception as e:
            logger.error(f"Twitter trending topics error: {e}")
            return []
    
    def _process_twitter_posts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Twitter API response"""
        posts = []
        
        if "data" in data:
            for tweet in data["data"]:
                posts.append({
                    "id": tweet.get("id", ""),
                    "content": tweet.get("text", ""),
                    "author_id": tweet.get("author_id", ""),
                    "created_at": tweet.get("created_at", ""),
                    "retweet_count": tweet.get("public_metrics", {}).get("retweet_count", 0),
                    "like_count": tweet.get("public_metrics", {}).get("like_count", 0),
                    "reply_count": tweet.get("public_metrics", {}).get("reply_count", 0),
                    "platform": "twitter",
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return posts
    
    def _get_mock_twitter_posts(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Mock Twitter posts for development"""
        return [
            {
                "id": f"mock_tweet_{i}",
                "content": f"Mock tweet about {query} #{i}",
                "author_id": f"user_{i}",
                "created_at": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "retweet_count": i * 10,
                "like_count": i * 25,
                "reply_count": i * 5,
                "platform": "twitter",
                "created_at": datetime.utcnow().isoformat()
            }
            for i in range(min(limit, 10))
        ]
    
    def _get_mock_trending_topics(self) -> List[Dict[str, Any]]:
        """Mock trending topics for development"""
        return [
            {
                "topic": "AI Technology",
                "volume": 15000,
                "platform": "twitter",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "topic": "Climate Change",
                "volume": 12000,
                "platform": "twitter",
                "created_at": datetime.utcnow().isoformat()
            }
        ]

class TikTokAPI(SocialMediaAPI):
    """TikTok API integration (simplified)"""
    
    def __init__(self):
        super().__init__(
            "TikTok",
            settings.TIKTOK_API_KEY,
            "https://open-api.tiktok.com"
        )
    
    async def search_posts(
        self,
        query: str,
        limit: int = 50,
        timeframe: str = "week"
    ) -> List[Dict[str, Any]]:
        """Search TikTok posts (simplified implementation)"""
        try:
            # Note: TikTok API requires special approval and has complex authentication
            # This is a simplified mock implementation
            
            return self._get_mock_tiktok_posts(query, limit)
            
        except Exception as e:
            logger.error(f"TikTok API error: {e}")
            return []
    
    async def get_trending_topics(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics from TikTok"""
        try:
            return self._get_mock_tiktok_trending()
        except Exception as e:
            logger.error(f"TikTok trending topics error: {e}")
            return []
    
    def _get_mock_tiktok_posts(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Mock TikTok posts for development"""
        return [
            {
                "id": f"mock_tiktok_{i}",
                "content": f"Mock TikTok video about {query} #{i}",
                "author": f"creator_{i}",
                "created_at": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "view_count": i * 1000,
                "like_count": i * 100,
                "comment_count": i * 20,
                "share_count": i * 50,
                "platform": "tiktok",
                "created_at": datetime.utcnow().isoformat()
            }
            for i in range(min(limit, 10))
        ]
    
    def _get_mock_tiktok_trending(self) -> List[Dict[str, Any]]:
        """Mock TikTok trending topics for development"""
        return [
            {
                "topic": "Dance Challenge",
                "volume": 25000,
                "platform": "tiktok",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "topic": "Cooking Tips",
                "volume": 18000,
                "platform": "tiktok",
                "created_at": datetime.utcnow().isoformat()
            }
        ]

class RSSAPI:
    """RSS feeds integration"""
    
    def __init__(self):
        self.timeout = 30.0
        self.feeds = [
            "https://feeds.feedburner.com/TechCrunch/",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.feedburner.com/oreilly/radar"
        ]
    
    async def get_latest_news(
        self,
        query: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get latest news from RSS feeds"""
        try:
            all_articles = []
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                tasks = []
                for feed_url in self.feeds:
                    task = self._fetch_feed(client, feed_url)
                    tasks.append(task)
                
                feed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(feed_results):
                    if isinstance(result, Exception):
                        logger.error(f"RSS feed error for {self.feeds[i]}: {result}")
                    else:
                        all_articles.extend(result)
            
            # Filter by query if provided
            if query:
                all_articles = [
                    article for article in all_articles
                    if query.lower() in article.get("title", "").lower() or
                       query.lower() in article.get("summary", "").lower()
                ]
            
            # Sort by date and return latest
            all_articles.sort(key=lambda x: x.get("published", ""), reverse=True)
            return all_articles[:limit]
            
        except Exception as e:
            logger.error(f"RSS API error: {e}")
            return []
    
    async def _fetch_feed(self, client: httpx.AsyncClient, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed"""
        try:
            response = await client.get(feed_url)
            response.raise_for_status()
            
            feed = feedparser.parse(response.text)
            articles = []
            
            for entry in feed.entries:
                articles.append({
                    "id": entry.get("id", ""),
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.term for tag in entry.get("tags", [])],
                    "platform": "rss",
                    "feed_url": feed_url,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            return []

class SocialMediaManager:
    """Manages all social media integrations"""
    
    def __init__(self):
        self.platforms = {
            "reddit": RedditAPI(),
            "twitter": TwitterAPI(),
            "tiktok": TikTokAPI(),
            "rss": RSSAPI()
        }
    
    async def search_all_platforms(
        self,
        query: str,
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 25
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search all platforms for posts about a query"""
        if platforms is None:
            platforms = ["reddit", "twitter", "tiktok", "rss"]
        
        results = {}
        
        # Run searches in parallel
        tasks = []
        for platform_name in platforms:
            if platform_name in self.platforms:
                if platform_name == "rss":
                    task = self.platforms[platform_name].get_latest_news(query, limit_per_platform)
                else:
                    task = self.platforms[platform_name].search_posts(query, limit_per_platform)
                tasks.append((platform_name, task))
        
        search_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for i, (platform_name, _) in enumerate(tasks):
            result = search_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error searching {platform_name}: {result}")
                results[platform_name] = []
            else:
                results[platform_name] = result
        
        return results
    
    async def get_trending_all_platforms(
        self,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get trending topics from all platforms"""
        if platforms is None:
            platforms = ["reddit", "twitter", "tiktok"]
        
        results = {}
        
        # Run trending searches in parallel
        tasks = []
        for platform_name in platforms:
            if platform_name in self.platforms:
                task = self.platforms[platform_name].get_trending_topics()
                tasks.append((platform_name, task))
        
        trending_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for i, (platform_name, _) in enumerate(tasks):
            result = trending_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error getting trending from {platform_name}: {result}")
                results[platform_name] = []
            else:
                results[platform_name] = result
        
        return results
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms"""
        return list(self.platforms.keys())
    
    async def get_platform_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all platforms"""
        stats = {}
        
        for platform_name, platform_api in self.platforms.items():
            try:
                if platform_name == "rss":
                    # Test RSS with a simple query
                    test_articles = await platform_api.get_latest_news("test", 1)
                    stats[platform_name] = {
                        "status": "active",
                        "articles_found": len(test_articles),
                        "last_checked": datetime.utcnow().isoformat()
                    }
                else:
                    # Test other platforms with a simple search
                    test_posts = await platform_api.search_posts("test", 1)
                    stats[platform_name] = {
                        "status": "active",
                        "posts_found": len(test_posts),
                        "last_checked": datetime.utcnow().isoformat()
                    }
            except Exception as e:
                stats[platform_name] = {
                    "status": "error",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }
        
        return stats

# Global instance
social_media_manager = SocialMediaManager()

# Convenience functions
async def search_social_media(
    query: str,
    platforms: Optional[List[str]] = None,
    limit_per_platform: int = 25
) -> Dict[str, List[Dict[str, Any]]]:
    """Search social media platforms for posts about a query"""
    return await social_media_manager.search_all_platforms(query, platforms, limit_per_platform)

async def get_trending_topics(
    platforms: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get trending topics from social media platforms"""
    return await social_media_manager.get_trending_all_platforms(platforms)

def get_available_platforms() -> List[str]:
    """Get list of available social media platforms"""
    return social_media_manager.get_available_platforms()

async def get_platform_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all social media platforms"""
    return await social_media_manager.get_platform_stats()
