"""
ContentService for content generation and management
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
from ..models.content_ideas import ContentIdeas, ContentType, ContentAngle, ContentStatus
from ..models.trend_analysis import TrendAnalysis
from ..models.keyword_data import KeywordData

logger = structlog.get_logger()
settings = get_settings()

class ContentService:
    """Service for content generation and management"""
    
    def __init__(self):
        # API keys are now retrieved from database as needed
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.google_ai_api_key = None
        
        # Content optimization APIs
        self.surfer_seo_api_key = None
        self.frase_api_key = None
        self.coschedule_api_key = None
        
        # Model configuration - will be determined dynamically
        self.llm_model = "gpt-4"  # Default, will be updated based on available API keys
        self.max_ideas = 20
        self.content_types = [ContentType.ARTICLE, ContentType.GUIDE, ContentType.REVIEW, ContentType.TUTORIAL, ContentType.LISTICLE]
        self.content_angles = [ContentAngle.HOW_TO, ContentAngle.VS, ContentAngle.LISTICLE, ContentAngle.PAIN_POINT, ContentAngle.STORY]
    
    async def generate_ideas(self, user_id: int, trend_analysis_id: int, keyword_data_id: int,
                           content_types: List[str], max_ideas: int = 5) -> Dict[str, Any]:
        """Generate content ideas"""
        try:
            # Validate inputs
            if max_ideas > self.max_ideas:
                max_ideas = self.max_ideas
            
            # Get trend analysis and keyword data
            db = next(get_db())
            trend_analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == trend_analysis_id)
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not trend_analysis or not keyword_data:
                raise ValueError("Trend analysis or keyword data not found")
            
            # Create content ideas record
            content_ideas = ContentIdeas(
                user_id=user_id,
                keyword_data_id=keyword_data_id,
                content_ideas=[]
            )
            db.add(content_ideas)
            db.commit()
            db.refresh(content_ideas)
            
            # Start background generation
            asyncio.create_task(self._generate_content_ideas(
                content_ideas.id, trend_analysis, keyword_data, content_types, max_ideas
            ))
            
            logger.info("Content ideas generation initiated", content_ideas_id=content_ideas.id)
            return content_ideas.to_dict()
            
        except Exception as e:
            logger.error("Failed to generate content ideas", error=str(e))
            raise
    
    async def get_content_idea(self, content_idea_id: int) -> Dict[str, Any]:
        """Get content idea by ID"""
        try:
            db = next(get_db())
            content_idea = db.get_ContentIdeas_by_id(ContentIdeas.id == content_idea_id)
            
            if not content_idea:
                raise ValueError("Content idea not found")
            
            return content_idea.to_dict()
            
        except Exception as e:
            logger.error("Failed to get content idea", content_idea_id=content_idea_id, error=str(e))
            raise
    
    async def update_content_status(self, content_idea_id: int, status: str, 
                                  scheduled_date: Optional[datetime] = None, notes: str = None) -> bool:
        """Update content idea status"""
        try:
            db = next(get_db())
            content_idea = db.get_ContentIdeas_by_id(ContentIdeas.id == content_idea_id)
            
            if not content_idea:
                raise ValueError("Content idea not found")
            
            # Update status
            content_idea.update_status(ContentStatus(status), scheduled_date)
            
            if notes:
                content_idea.add_note(notes)
            
            db.commit()
            
            logger.info("Content status updated", content_idea_id=content_idea_id, status=status)
            return True
            
        except Exception as e:
            logger.error("Failed to update content status", content_idea_id=content_idea_id, error=str(e))
            raise
    
    async def _generate_content_ideas(self, content_ideas_id: int, trend_analysis: TrendAnalysis,
                                    keyword_data: KeywordData, content_types: List[str], max_ideas: int):
        """Generate content ideas in background"""
        try:
            db = next(get_db())
            content_ideas = db.get_ContentIdeas_by_id(ContentIdeas.id == content_ideas_id)
            
            if not content_ideas:
                return
            
            # Get high-opportunity topics
            high_opportunity_topics = trend_analysis.get_high_opportunity_topics(70.0)
            
            # Get top keywords
            top_keywords = keyword_data.get_top_keywords(20)
            
            # Generate ideas for each content type
            generated_ideas = []
            for content_type in content_types:
                if len(generated_ideas) >= max_ideas:
                    break
                
                ideas = await self._generate_ideas_for_type(
                    content_type, high_opportunity_topics, top_keywords, trend_analysis, keyword_data
                )
                generated_ideas.extend(ideas)
            
            # Limit to max_ideas
            generated_ideas = generated_ideas[:max_ideas]
            
            # Update content ideas
            content_ideas.content_ideas = generated_ideas
            db.commit()
            
            logger.info("Content ideas generation completed", 
                       content_ideas_id=content_ideas_id, 
                       ideas_count=len(generated_ideas))
            
        except Exception as e:
            logger.error("Content ideas generation failed", content_ideas_id=content_ideas_id, error=str(e))
    
    async def _generate_ideas_for_type(self, content_type: str, topics: List[str], 
                                     keywords: List[Dict[str, Any]], trend_analysis: TrendAnalysis,
                                     keyword_data: KeywordData) -> List[Dict[str, Any]]:
        """Generate ideas for specific content type"""
        ideas = []
        
        for topic in topics:
            # Generate ideas for each angle
            for angle in self.content_angles:
                idea = await self._generate_single_idea(
                    content_type, angle, topic, keywords, trend_analysis, keyword_data
                )
                if idea:
                    ideas.append(idea)
        
        return ideas
    
    async def _generate_single_idea(self, content_type: str, angle: str, topic: str,
                                  keywords: List[Dict[str, Any]], trend_analysis: TrendAnalysis,
                                  keyword_data: KeywordData) -> Optional[Dict[str, Any]]:
        """Generate single content idea"""
        try:
            # Get relevant keywords for topic
            topic_keywords = [k for k in keywords if topic.lower() in k.get("keyword", "").lower()]
            if not topic_keywords:
                topic_keywords = keywords[:5]  # Use top keywords as fallback
            
            # Generate title
            title = self._generate_title(content_type, angle, topic, topic_keywords)
            
            # Generate outline
            outline = self._generate_outline(content_type, angle, topic, topic_keywords)
            
            # Generate SEO recommendations
            seo_recommendations = self._generate_seo_recommendations(topic, topic_keywords)
            
            # Calculate scores
            headline_score = self._calculate_headline_score(title)
            priority_score = self._calculate_priority_score(topic, trend_analysis, keyword_data)
            
            # Generate affiliate opportunities
            affiliate_opportunities = self._generate_affiliate_opportunities(topic, topic_keywords)
            
            return {
                "id": f"idea_{hash(topic + angle + content_type)}",
                "title": title,
                "content_type": content_type,
                "angle": angle,
                "headline_score": headline_score,
                "priority_score": priority_score,
                "outline": outline,
                "seo_recommendations": seo_recommendations,
                "affiliate_opportunities": affiliate_opportunities,
                "status": "DRAFT"
            }
            
        except Exception as e:
            logger.error("Failed to generate single idea", error=str(e))
            return None
    
    def _generate_title(self, content_type: str, angle: str, topic: str, keywords: List[Dict[str, Any]]) -> str:
        """Generate content title"""
        # Get top keyword for title
        top_keyword = keywords[0]["keyword"] if keywords else topic
        
        # Generate title based on type and angle
        title_templates = {
            "article": {
                "how-to": f"How to {topic}: Complete Guide for {top_keyword}",
                "vs": f"{topic} vs Alternatives: Which is Best for {top_keyword}?",
                "listicle": f"10 Best {topic} Tips for {top_keyword} Success",
                "pain-point": f"Why {topic} is Failing (And How to Fix It)",
                "story": f"My {topic} Journey: Lessons Learned from {top_keyword}"
            },
            "guide": {
                "how-to": f"The Complete {topic} Guide: Everything You Need to Know",
                "vs": f"{topic} vs Competitors: Comprehensive Comparison",
                "listicle": f"Ultimate {topic} Checklist: 25 Essential Steps",
                "pain-point": f"Common {topic} Mistakes and How to Avoid Them",
                "story": f"From Beginner to Expert: My {topic} Success Story"
            },
            "review": {
                "how-to": f"How to Choose the Best {topic} for {top_keyword}",
                "vs": f"{topic} Comparison: Top 5 Options Reviewed",
                "listicle": f"Best {topic} Products: 10 Expert Reviews",
                "pain-point": f"Honest {topic} Review: What They Don't Tell You",
                "story": f"I Tested {topic} for 30 Days: Here's What Happened"
            },
            "tutorial": {
                "how-to": f"Step-by-Step {topic} Tutorial for Beginners",
                "vs": f"{topic} Methods Compared: Which Works Best?",
                "listicle": f"10 {topic} Techniques Every {top_keyword} Should Know",
                "pain-point": f"Fix Your {topic} Problems with This Tutorial",
                "story": f"Learning {topic}: My 30-Day Challenge Results"
            },
            "listicle": {
                "how-to": f"How to {topic}: 15 Essential Steps",
                "vs": f"{topic} vs Alternatives: 10 Key Differences",
                "listicle": f"25 {topic} Tips for {top_keyword} Success",
                "pain-point": f"15 {topic} Mistakes to Avoid",
                "story": f"10 {topic} Lessons I Learned the Hard Way"
            }
        }
        
        template = title_templates.get(content_type, {}).get(angle, f"{topic}: {angle.title()} Guide")
        return template
    
    def _generate_outline(self, content_type: str, angle: str, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content outline"""
        # Get top 3 keywords for outline
        top_keywords = [k["keyword"] for k in keywords[:3]]
        
        outline = {
            "introduction": f"Introduction to {topic} and why it matters for {top_keywords[0] if top_keywords else 'your success'}",
            "sections": []
        }
        
        # Generate sections based on content type and angle
        if content_type == "guide":
            sections = [
                {
                    "title": f"Understanding {topic}",
                    "subsections": [
                        f"What is {topic}?",
                        f"Benefits of {topic}",
                        f"Common misconceptions about {topic}"
                    ]
                },
                {
                    "title": f"Getting Started with {topic}",
                    "subsections": [
                        f"Prerequisites for {topic}",
                        f"Step-by-step {topic} process",
                        f"Tools and resources needed"
                    ]
                },
                {
                    "title": f"Advanced {topic} Techniques",
                    "subsections": [
                        f"Pro tips for {topic}",
                        f"Common {topic} challenges",
                        f"Optimizing your {topic} results"
                    ]
                }
            ]
        elif content_type == "review":
            sections = [
                {
                    "title": f"{topic} Overview",
                    "subsections": [
                        f"What is {topic}?",
                        f"Key features of {topic}",
                        f"Who should use {topic}?"
                    ]
                },
                {
                    "title": f"{topic} Pros and Cons",
                    "subsections": [
                        f"Advantages of {topic}",
                        f"Disadvantages of {topic}",
                        f"Comparison with alternatives"
                    ]
                },
                {
                    "title": f"{topic} Verdict",
                    "subsections": [
                        f"Is {topic} worth it?",
                        f"Best use cases for {topic}",
                        f"Final recommendation"
                    ]
                }
            ]
        else:  # article, tutorial, listicle
            sections = [
                {
                    "title": f"Introduction to {topic}",
                    "subsections": [
                        f"What you'll learn about {topic}",
                        f"Why {topic} matters",
                        f"Prerequisites for this {content_type}"
                    ]
                },
                {
                    "title": f"Main {topic} Content",
                    "subsections": [
                        f"Step 1: Understanding {topic}",
                        f"Step 2: Implementing {topic}",
                        f"Step 3: Optimizing {topic}"
                    ]
                },
                {
                    "title": f"Conclusion and Next Steps",
                    "subsections": [
                        f"Key takeaways about {topic}",
                        f"Next steps for {topic}",
                        f"Additional resources for {topic}"
                    ]
                }
            ]
        
        outline["sections"] = sections
        outline["conclusion"] = f"Summary of {topic} and actionable next steps"
        
        return outline
    
    def _generate_seo_recommendations(self, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate SEO recommendations"""
        # Get target keywords
        target_keywords = [k["keyword"] for k in keywords[:5]]
        
        # Generate meta description
        meta_description = f"Learn everything about {topic} with our comprehensive guide. Discover {target_keywords[0] if target_keywords else 'expert tips'} and strategies for success."
        
        # Generate heading structure
        heading_structure = "H1: Main title, H2: Section titles, H3: Subsection titles"
        
        # Calculate word count target
        word_count_target = 2000 + (len(topic) * 10)  # Base 2000 + topic length factor
        
        # Generate internal links
        internal_links = [
            f"{topic} guide",
            f"{topic} tips",
            f"{topic} examples",
            f"related to {topic}"
        ]
        
        # Generate external links
        external_links = [
            f"{topic} official documentation",
            f"{topic} best practices",
            f"{topic} community forum"
        ]
        
        return {
            "target_keywords": target_keywords,
            "meta_description": meta_description,
            "heading_structure": heading_structure,
            "word_count_target": word_count_target,
            "internal_links": internal_links,
            "external_links": external_links
        }
    
    def _calculate_headline_score(self, title: str) -> float:
        """Calculate headline score using CoSchedule-like scoring"""
        score = 50.0  # Base score
        
        # Length scoring (optimal 6-12 words)
        word_count = len(title.split())
        if 6 <= word_count <= 12:
            score += 20
        elif 4 <= word_count <= 15:
            score += 10
        
        # Power words
        power_words = ["complete", "ultimate", "best", "guide", "how", "why", "what", "when", "where"]
        power_word_count = sum(1 for word in power_words if word.lower() in title.lower())
        score += power_word_count * 5
        
        # Emotional triggers
        emotional_words = ["amazing", "incredible", "shocking", "secret", "proven", "expert"]
        emotional_count = sum(1 for word in emotional_words if word.lower() in title.lower())
        score += emotional_count * 3
        
        # Numbers
        if any(char.isdigit() for char in title):
            score += 10
        
        # Question format
        if title.endswith("?"):
            score += 5
        
        return min(max(score, 0), 100)
    
    def _calculate_priority_score(self, topic: str, trend_analysis: TrendAnalysis, keyword_data: KeywordData) -> float:
        """Calculate priority score for content idea"""
        # Get opportunity score from trend analysis
        opportunity_score = trend_analysis.get_opportunity_score(topic) / 100.0
        
        # Get keyword priority scores
        topic_keywords = [k for k in keyword_data.keywords if topic.lower() in k.get("keyword", "").lower()]
        if topic_keywords:
            avg_keyword_priority = sum(k.get("priority_score", 0) for k in topic_keywords) / len(topic_keywords)
        else:
            avg_keyword_priority = 0.5
        
        # Weighted average
        priority_score = (opportunity_score * 0.6 + avg_keyword_priority * 0.4)
        
        return min(max(priority_score, 0), 1)
    
    def _generate_affiliate_opportunities(self, topic: str, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate affiliate opportunities"""
        opportunities = []
        
        # Generate opportunities based on topic and keywords
        if "coffee" in topic.lower():
            opportunities.extend([
                {
                    "program": "Coffee Equipment Pro",
                    "product": "Coffee Roasters",
                    "placement": "equipment section",
                    "context": "When discussing coffee roasting equipment"
                },
                {
                    "program": "Amazon Associates",
                    "product": "Coffee Beans",
                    "placement": "product recommendations",
                    "context": "When recommending coffee beans"
                }
            ])
        elif "software" in topic.lower():
            opportunities.extend([
                {
                    "program": "Software Tools Pro",
                    "product": "Development Tools",
                    "placement": "tools section",
                    "context": "When discussing software development tools"
                }
            ])
        else:
            # Generic opportunities
            opportunities.append({
                "program": "General Affiliate",
                "product": f"{topic} Products",
                "placement": "product section",
                "context": f"When discussing {topic} products"
            })
        
        return opportunities
    
    async def get_user_content_ideas(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's content ideas"""
        try:
            db = next(get_db())
            content_ideas = db.query(ContentIdeas).filter(
                ContentIdeas.user_id == user_id
            ).order_by(ContentIdeas.created_at.desc()).offset(offset).limit(limit).all()
            
            return [ci.to_dict() for ci in content_ideas]
            
        except Exception as e:
            logger.error("Failed to get user content ideas", user_id=user_id, error=str(e))
            raise
    
    async def delete_content_idea(self, content_idea_id: int, user_id: int) -> bool:
        """Delete content idea"""
        try:
            db = next(get_db())
            content_idea = db.get_ContentIdeas_by_id(
                ContentIdeas.id == content_idea_id,
                ContentIdeas.user_id == user_id
            )
            
            if not content_idea:
                raise ValueError("Content idea not found")
            
            db.delete(content_idea)
            db.commit()
            
            logger.info("Content idea deleted", content_idea_id=content_idea_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete content idea", content_idea_id=content_idea_id, error=str(e))
            raise
