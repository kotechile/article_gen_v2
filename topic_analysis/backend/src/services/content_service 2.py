"""
ContentService for content generation and management
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache
from ..core.config import settings
from ..schemas.content_schemas import ContentType, ContentStatus
# Models are now handled by Supabase directly
# from ..models.content_ideas import ContentIdeas, ContentType, ContentAngle, ContentStatus
# from ..models.trend_analysis import TrendAnalysis
# from ..models.keyword_data import KeywordData

# Define ContentAngle enum locally since it's not in the schemas
from enum import Enum

class ContentAngle(str, Enum):
    """Content angle enumeration"""
    HOW_TO = "how-to"
    VS = "vs"
    LISTICLE = "listicle"
    PAIN_POINT = "pain-point"
    STORY = "story"

logger = structlog.get_logger()

class ContentService:
    """Service for content generation and management"""
    
    def __init__(self):
        self.openai_api_key = settings.openai_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self.google_ai_api_key = settings.google_ai_api_key
        
        # Content optimization APIs
        self.surfer_seo_api_key = settings.surfer_seo_api_key
        self.frase_api_key = settings.frase_api_key
        self.coschedule_api_key = settings.coschedule_api_key
        
        # Model configuration
        self.llm_model = "gpt-4" if self.openai_api_key else "claude-3-sonnet"
        self.max_ideas = 20
        self.content_types = [ContentType.ARTICLE, ContentType.HOW_TO_GUIDE, ContentType.PRODUCT_REVIEW, ContentType.BLOG_POST, ContentType.SOCIAL_MEDIA]
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
            # trend_analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == trend_analysis_id)
            # Using Supabase directly instead of SQLAlchemy models
            trend_analysis = None  # Will be fetched from Supabase
            # keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            # Using Supabase directly instead of SQLAlchemy models
            keyword_data = None  # Will be fetched from Supabase
            
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
    
    async def _generate_content_ideas(self, content_ideas_id: int, trend_analysis: Dict[str, Any],
                                    keyword_data: Dict[str, Any], content_types: List[str], max_ideas: int):
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
                                     keywords: List[Dict[str, Any]], trend_analysis: Dict[str, Any],
                                     keyword_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                                  keywords: List[Dict[str, Any]], trend_analysis: Dict[str, Any],
                                  keyword_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    
    def _calculate_priority_score(self, topic: str, trend_analysis: Dict[str, Any], keyword_data: Dict[str, Any]) -> float:
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
    
    async def generate_content_ideas(
        self,
        search_term: str,
        trending_topics: List[Dict[str, Any]],
        affiliate_programs: List[Dict[str, Any]] = None,
        content_type: str = "blog",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate high-quality content ideas using LLM based on trends and search terms
        """
        try:
            logger.info("Generating content ideas with LLM", 
                       search_term=search_term, 
                       trending_topics=len(trending_topics),
                       content_type=content_type)
            
            # Prepare context for LLM
            trends_context = self._prepare_trends_context(trending_topics)
            affiliate_context = self._prepare_affiliate_context(affiliate_programs) if affiliate_programs else ""
            
            # Create a comprehensive LLM prompt for content generation
            prompt = self._create_content_generation_prompt(
                search_term, trends_context, affiliate_context, content_type
            )
            
            # Call LLM to generate content ideas
            content_ideas = await self._call_llm_for_content_ideas(prompt)
            
            # Process and enhance the generated ideas
            processed_ideas = self._process_generated_ideas(content_ideas, search_term, trending_topics)
            
            logger.info("Content ideas generated successfully", 
                       ideas_count=len(processed_ideas),
                       search_term=search_term)
            
            return {
                "content_ideas": processed_ideas,
                "total_ideas": len(processed_ideas),
                "search_term": search_term,
                "trending_topics_used": len(trending_topics),
                "affiliate_programs_used": len(affiliate_programs) if affiliate_programs else 0,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Content ideas generation failed", error=str(e))
            # Return fallback ideas if LLM fails
            return self._generate_fallback_content_ideas(search_term, trending_topics)
    
    def _prepare_trends_context(self, trending_topics: List[Dict[str, Any]]) -> str:
        """Prepare trends context for LLM prompt"""
        if not trending_topics:
            return "No specific trends provided"
        
        context_parts = []
        for topic in trending_topics[:5]:  # Limit to top 5 trends
            topic_name = topic.get('topic', 'Unknown Topic')
            search_volume = topic.get('search_volume', 0)
            trend_direction = topic.get('trend_direction', 'stable')
            opportunity_score = topic.get('opportunity_score', 0)
            
            context_parts.append(
                f"- {topic_name}: {search_volume:,} monthly searches, {trend_direction} trend, {opportunity_score}% opportunity score"
            )
        
        return "\n".join(context_parts)
    
    def _prepare_affiliate_context(self, affiliate_programs: List[Dict[str, Any]]) -> str:
        """Prepare affiliate programs context for LLM prompt"""
        if not affiliate_programs:
            return "No affiliate programs specified"
        
        context_parts = []
        for program in affiliate_programs[:3]:  # Limit to top 3 programs
            name = program.get('name', 'Unknown Program')
            commission = program.get('commission', 'Unknown')
            context_parts.append(f"- {name}: {commission} commission")
        
        return "\n".join(context_parts)
    
    def _create_content_generation_prompt(
        self, 
        search_term: str, 
        trends_context: str, 
        affiliate_context: str, 
        content_type: str
    ) -> str:
        """Create a comprehensive LLM prompt for content generation"""
        
        return f"""
You are an expert content strategist and SEO specialist. Generate 8-12 high-quality, specific content ideas for the search term "{search_term}".

CONTEXT:
Search Term: {search_term}
Content Type: {content_type}
Target Audience: People interested in {search_term}

TRENDING TOPICS:
{trends_context}

AFFILIATE OPPORTUNITIES:
{affiliate_context}

REQUIREMENTS:
1. Create SPECIFIC, ACTIONABLE content ideas (not generic templates)
2. Each idea should be unique and valuable to readers
3. Focus on solving real problems and providing genuine value
4. Include specific angles, formats, and target audiences
5. Make titles compelling and click-worthy
6. Consider seasonal trends and current events
7. Include both beginner and advanced level content
8. Mix different content formats including:
   - BLOG CONTENT: guides, reviews, comparisons, tutorials, listicles, case studies
   - SOFTWARE IDEAS: web applications, mobile apps, SaaS tools, calculator tools, dashboard apps, community platforms, marketplace apps, booking systems, analytics tools, content generators

CONTENT IDEA FORMAT:
For each idea, provide:
- title: Specific, compelling title (60 characters or less)
- description: 2-3 sentence description explaining the value
- content_type: Specific format (e.g., "comprehensive guide", "product comparison", "step-by-step tutorial", "mobile app", "web application", "SaaS tool", "calculator tool", "dashboard app")
- target_audience: Who this is for (e.g., "beginner runners", "fitness enthusiasts", "tech professionals")
- difficulty: "Beginner", "Intermediate", or "Advanced"
- estimated_time: Realistic time estimate (e.g., "2-4 hours", "1-2 days", "1 week")
- key_benefits: 3-4 specific benefits readers will get
- target_keywords: 5-8 relevant keywords for SEO

EXAMPLES OF GOOD IDEAS:
BLOG CONTENT:
- "Best Running Shoes for Flat Feet: 2024 Complete Guide" (not "Running Shoes Complete Guide")
- "How to Choose the Right Protein Powder for Muscle Building" (not "Protein Powder Guide")
- "Nike vs Adidas Running Shoes: Which Brand Wins in 2024?" (not "Running Shoes Comparison")

SOFTWARE IDEAS:
- "Fitness Tracker Mobile App for Beginners" (not "Fitness App")
- "Running Pace Calculator Web Tool" (not "Calculator Tool")
- "Gym Management SaaS Platform" (not "Management Tool")
- "Fitness Community Dashboard" (not "Dashboard App")

EXAMPLES OF BAD IDEAS (avoid these):
- "Running Shoes Complete Guide: Advanced Level Guide"
- "Running Apps and Technology Complete Guide: Advanced Level Guide"
- Generic templates that just replace the topic name

Generate 8-12 specific, valuable content ideas that people would actually want to read and share. 
IMPORTANT: Include a mix of both blog content ideas AND software/app ideas. 
Aim for approximately 60% blog content and 40% software ideas to provide a balanced content strategy.

Return as JSON array with this exact structure:
[
  {{
    "title": "Specific, compelling title",
    "description": "2-3 sentence description of value",
    "content_type": "specific format",
    "target_audience": "specific audience",
    "difficulty": "Beginner/Intermediate/Advanced",
    "estimated_time": "realistic time estimate",
    "key_benefits": ["benefit 1", "benefit 2", "benefit 3"],
    "target_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
  }}
]
"""
    
    async def _call_llm_for_content_ideas(self, prompt: str) -> List[Dict[str, Any]]:
        """Call LLM to generate content ideas"""
        try:
            from ..integrations.llm_providers import generate_content
            
            # Call the LLM
            result = await generate_content(
                prompt=prompt,
                provider="openai" if self.openai_api_key else "anthropic",
                max_tokens=2000,
                temperature=0.7
            )
            
            if "error" in result:
                logger.warning("LLM call failed, using fallback", error=result["error"])
                return []
            
            # Parse the JSON response
            content_text = result.get("content", "")
            if not content_text:
                return []
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content_text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
            else:
                logger.warning("No JSON found in LLM response")
                return []
                
        except Exception as e:
            logger.error("LLM call failed", error=str(e))
            return []
    
    def _process_generated_ideas(
        self, 
        content_ideas: List[Dict[str, Any]], 
        search_term: str, 
        trending_topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process and enhance generated content ideas"""
        processed_ideas = []
        
        for i, idea in enumerate(content_ideas):
            # Add required fields and enhance the idea
            processed_idea = {
                "id": f"idea_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i}",
                "title": idea.get("title", f"Content Idea {i+1}"),
                "description": idea.get("description", "AI-generated content idea"),
                "content_type": idea.get("content_type", "blog_post"),
                "target_audience": idea.get("target_audience", "general audience"),
                "difficulty": idea.get("difficulty", "Intermediate"),
                "estimated_time": idea.get("estimated_time", "2-4 hours"),
                "key_benefits": idea.get("key_benefits", []),
                "target_keywords": idea.get("target_keywords", []),
                "trend_score": self._calculate_trend_score(idea, trending_topics),
                "opportunity_score": self._calculate_opportunity_score(idea, trending_topics),
                "base_topic": search_term,
                "enhanced_keywords": [],
                "seo_optimized": False,
                "created_at": datetime.utcnow().isoformat()
            }
            
            processed_ideas.append(processed_idea)
        
        return processed_ideas
    
    def _calculate_trend_score(self, idea: Dict[str, Any], trending_topics: List[Dict[str, Any]]) -> int:
        """Calculate trend score for an idea"""
        # Base score
        score = 60
        
        # Check if idea title contains trending topics
        title = idea.get("title", "").lower()
        for topic in trending_topics:
            topic_name = topic.get("topic", "").lower()
            if topic_name in title:
                score += 20
                break
        
        # Add some randomness for variety
        import random
        score += random.randint(-10, 20)
        
        return min(max(score, 40), 100)
    
    def _calculate_opportunity_score(self, idea: Dict[str, Any], trending_topics: List[Dict[str, Any]]) -> int:
        """Calculate opportunity score for an idea"""
        # Base score
        score = 70
        
        # Check if idea has specific keywords
        keywords = idea.get("target_keywords", [])
        if len(keywords) >= 5:
            score += 15
        
        # Check if idea is specific (not generic)
        title = idea.get("title", "")
        if any(word in title.lower() for word in ["best", "how to", "vs", "review", "guide", "tips"]):
            score += 10
        
        # Add some randomness for variety
        import random
        score += random.randint(-15, 15)
        
        return min(max(score, 50), 100)
    
    def _generate_fallback_content_ideas(
        self, 
        search_term: str, 
        trending_topics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate fallback content ideas if LLM fails"""
        fallback_ideas = [
            {
                "id": f"fallback_idea_{i}",
                "title": f"Complete Guide to {search_term}",
                "description": f"Everything you need to know about {search_term}",
                "content_type": "comprehensive guide",
                "target_audience": "beginners",
                "difficulty": "Beginner",
                "estimated_time": "2-4 hours",
                "key_benefits": ["Learn the basics", "Get started quickly", "Avoid common mistakes"],
                "target_keywords": [search_term, f"{search_term} guide", f"how to {search_term}"],
                "trend_score": 75,
                "opportunity_score": 80,
                "base_topic": search_term,
                "enhanced_keywords": [],
                "seo_optimized": False,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return {
            "content_ideas": fallback_ideas,
            "total_ideas": len(fallback_ideas),
            "search_term": search_term,
            "trending_topics_used": len(trending_topics),
            "affiliate_programs_used": 0,
            "generated_at": datetime.utcnow().isoformat()
        }