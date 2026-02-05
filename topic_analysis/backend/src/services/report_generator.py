"""
Report generator service for creating keyword analysis reports
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from ..models.analysis_report import KeywordAnalysisReport
from ..models.keyword import Keyword
from ..config import settings
from .database import DatabaseService

logger = logging.getLogger(__name__)

class ReportGeneratorService:
    """Service for generating keyword analysis reports"""
    
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service or DatabaseService()
        self.report_template = {
            'summary': {},
            'top_opportunities': {},
            'content_recommendations': [],
            'insights': [],
            'next_steps': [],
            'seo_content_ideas': []
        }
    
    def generate_report(
        self, 
        keywords: List[Dict[str, Any]], 
        file_id: str, 
        user_id: str,
        analysis_id: str,
        scoring_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive keyword analysis report
        
        Args:
            keywords: List of analyzed keywords
            file_id: ID of the uploaded file
            user_id: ID of the user
            analysis_id: ID of the analysis
            scoring_weights: Scoring weights used for analysis
            
        Returns:
            Complete analysis report
        """
        try:
            # Generate report sections
            summary = self._generate_summary(keywords)
            top_opportunities = self._get_top_opportunities(keywords)
            content_recommendations = self._get_content_recommendations(keywords)
            insights = self._generate_insights(keywords)
            next_steps = self._generate_next_steps(keywords)
            seo_content_ideas = self._generate_seo_content_ideas(keywords)
            
            # Create report
            report = {
                'report_id': analysis_id,
                'file_id': file_id,
                'user_id': user_id,
                'summary': summary,
                'top_opportunities': top_opportunities,
                'content_recommendations': content_recommendations,
                'insights': insights,
                'next_steps': next_steps,
                'seo_content_ideas': seo_content_ideas,
                'generated_at': datetime.utcnow().isoformat(),
                'scoring_weights': scoring_weights or {}
            }
            
            logger.info(f"Generated report for {len(keywords)} keywords")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise ValueError(f"Error generating report: {str(e)}")
    
    def _generate_summary(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not keywords:
            return {}
        
        # Calculate totals
        total_keywords = len(keywords)
        total_search_volume = sum(k.get('Volume', 0) for k in keywords)
        
        # Count by category
        high_count = len([k for k in keywords if k.get('category') == 'high'])
        medium_count = len([k for k in keywords if k.get('category') == 'medium'])
        low_count = len([k for k in keywords if k.get('category') == 'low'])
        
        # Calculate averages
        avg_difficulty = sum(k.get('Difficulty', 0) for k in keywords) / total_keywords
        avg_cpc = sum(k.get('CPC', 0) for k in keywords) / total_keywords
        
        return {
            'total_keywords': total_keywords,
            'high_opportunity_count': high_count,
            'medium_opportunity_count': medium_count,
            'low_opportunity_count': low_count,
            'total_search_volume': total_search_volume,
            'average_difficulty': round(avg_difficulty, 2),
            'average_cpc': round(avg_cpc, 2)
        }
    
    def _get_top_opportunities(self, keywords: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Get top opportunities from keywords"""
        
        # High opportunity keywords (top 10)
        high_opportunity = sorted(
            [k for k in keywords if k.get('category') == 'high'],
            key=lambda x: x.get('opportunity_score', 0),
            reverse=True
        )[:10]
        
        # Quick wins (low difficulty, decent volume)
        quick_wins = sorted(
            [k for k in keywords if k.get('Difficulty', 0) <= 25 and k.get('Volume', 0) >= 200],
            key=lambda x: x.get('opportunity_score', 0),
            reverse=True
        )[:10]
        
        # High volume targets (top 5 by search volume)
        high_volume = sorted(
            keywords,
            key=lambda x: x.get('Volume', 0),
            reverse=True
        )[:5]
        
        return {
            'high_opportunity_keywords': high_opportunity,
            'quick_wins': quick_wins,
            'high_volume_targets': high_volume
        }
    
    def _get_content_recommendations(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get content format recommendations"""
        
        recommendations = []
        
        for keyword in keywords:
            keyword_text = keyword.get('Keyword', '').lower()
            content_format = self._detect_content_format(keyword_text)
            
            if content_format:
                recommendations.append({
                    'keyword': keyword.get('Keyword', ''),
                    'content_format': content_format,
                    'seo_score': keyword.get('opportunity_score', 0)
                })
        
        # Sort by SEO score
        recommendations.sort(key=lambda x: x['seo_score'], reverse=True)
        
        return recommendations
    
    def _detect_content_format(self, keyword: str) -> str:
        """Detect content format based on keyword patterns"""
        
        # List article patterns
        if any(pattern in keyword for pattern in ['best', 'top', 'list', 'review']):
            return 'list-article'
        
        # How-to guide patterns
        if any(pattern in keyword for pattern in ['how to', 'how do', 'guide', 'tutorial']):
            return 'how-to-guide'
        
        # Comparison patterns
        if any(pattern in keyword for pattern in ['vs', 'versus', 'compare', 'comparison']):
            return 'comparison'
        
        # Review patterns
        if any(pattern in keyword for pattern in ['review', 'rating', 'opinion']):
            return 'review'
        
        # Tutorial patterns
        if any(pattern in keyword for pattern in ['tutorial', 'learn', 'course']):
            return 'tutorial'
        
        # Case study patterns
        if any(pattern in keyword for pattern in ['case study', 'example', 'success story']):
            return 'case-study'
        
        # Default to informational
        return 'informational'
    
    def _generate_insights(self, keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from keyword analysis"""
        
        insights = []
        
        # High opportunity keywords
        high_opportunity_count = len([k for k in keywords if k.get('category') == 'high'])
        if high_opportunity_count > 0:
            insights.append(f"{high_opportunity_count} high-opportunity keywords identified")
        
        # Quick wins
        quick_wins_count = len([k for k in keywords if k.get('Difficulty', 0) <= 25 and k.get('Volume', 0) >= 200])
        if quick_wins_count > 0:
            insights.append(f"{quick_wins_count} quick-win keywords available (low difficulty, decent volume)")
        
        # High volume targets
        high_volume_count = len([k for k in keywords if k.get('Volume', 0) >= 5000])
        if high_volume_count > 0:
            insights.append(f"{high_volume_count} high-volume keywords for pillar content")
        
        # Commercial opportunities
        commercial_count = len([k for k in keywords if k.get('CPC', 0) >= 2.0])
        if commercial_count > 0:
            insights.append(f"{commercial_count} high-CPC keywords for monetization")
        
        # Intent distribution
        informational_count = len([k for k in keywords if k.get('primary_intent') == 'Informational'])
        if informational_count > 0:
            insights.append(f"{informational_count} informational keywords ideal for blog content")
        
        return insights
    
    def _generate_next_steps(self, keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate next steps based on keyword analysis"""
        
        next_steps = []
        
        # High opportunity keywords
        high_opportunity = [k for k in keywords if k.get('category') == 'high']
        if len(high_opportunity) > 0:
            next_steps.append("Prioritize high-opportunity keywords for immediate content creation")
        
        # Quick wins
        quick_wins = [k for k in keywords if k.get('Difficulty', 0) <= 25 and k.get('Volume', 0) >= 200]
        if len(quick_wins) > 0:
            next_steps.append("Create quick-win content for low-difficulty keywords")
        
        # High volume targets
        high_volume = [k for k in keywords if k.get('Volume', 0) >= 5000]
        if len(high_volume) > 0:
            next_steps.append("Develop pillar content around high-volume keywords")
        
        # Commercial opportunities
        commercial = [k for k in keywords if k.get('CPC', 0) >= 2.0]
        if len(commercial) > 0:
            next_steps.append("Focus on high-CPC keywords for monetization")
        
        # Content calendar
        next_steps.append("Create content calendar based on keyword priorities")
        
        return next_steps
    
    def _generate_seo_content_ideas(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate SEO content ideas from keywords"""
        
        content_ideas = []
        
        # Group keywords by topic
        topic_groups = self._group_keywords_by_topic(keywords)
        
        for topic, topic_keywords in topic_groups.items():
            if len(topic_keywords) >= 3:  # Need at least 3 keywords for a content idea
                content_idea = self._create_content_idea(topic, topic_keywords)
                if content_idea:
                    content_ideas.append(content_idea)
        
        # Sort by combined score
        content_ideas.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return content_ideas
    
    def _group_keywords_by_topic(self, keywords: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group keywords by topic for content ideas"""
        
        topic_groups = {}
        
        for keyword in keywords:
            keyword_text = keyword.get('Keyword', '').lower()
            
            # Simple topic detection based on common words
            topic = self._detect_topic(keyword_text)
            
            if topic not in topic_groups:
                topic_groups[topic] = []
            
            topic_groups[topic].append(keyword)
        
        return topic_groups
    
    def _detect_topic(self, keyword: str) -> str:
        """Detect topic from keyword"""
        
        # Common topic patterns
        topic_patterns = {
            'project_management': ['project', 'management', 'planning', 'tracking'],
            'software_tools': ['software', 'tool', 'app', 'platform'],
            'marketing': ['marketing', 'promotion', 'advertising', 'campaign'],
            'business': ['business', 'company', 'enterprise', 'corporate'],
            'technology': ['tech', 'digital', 'online', 'cloud'],
            'productivity': ['productivity', 'efficiency', 'workflow', 'automation']
        }
        
        for topic, patterns in topic_patterns.items():
            if any(pattern in keyword for pattern in patterns):
                return topic
        
        return 'general'
    
    def _create_content_idea(self, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a content idea from topic keywords"""
        
        if len(keywords) < 3:
            return None
        
        # Sort keywords by opportunity score
        sorted_keywords = sorted(keywords, key=lambda x: x.get('opportunity_score', 0), reverse=True)
        
        # Get primary and secondary keywords
        primary_keywords = [k.get('Keyword', '') for k in sorted_keywords[:3]]
        secondary_keywords = [k.get('Keyword', '') for k in sorted_keywords[3:6]]
        
        # Calculate metrics
        total_search_volume = sum(k.get('Volume', 0) for k in keywords)
        avg_difficulty = sum(k.get('Difficulty', 0) for k in keywords) / len(keywords)
        avg_cpc = sum(k.get('CPC', 0) for k in keywords) / len(keywords)
        
        # Generate title
        title = self._generate_content_title(topic, primary_keywords)
        
        # Determine content type
        content_type = self._determine_content_type(primary_keywords)
        
        # Calculate scores
        seo_score = self._calculate_seo_score(keywords)
        traffic_score = self._calculate_traffic_score(keywords)
        combined_score = (seo_score + traffic_score) / 2
        
        # Generate optimization tips
        optimization_tips = self._generate_optimization_tips(primary_keywords, secondary_keywords)
        
        # Generate content outline
        content_outline = self._generate_content_outline(content_type, primary_keywords)
        
        return {
            'id': f"idea_{topic}_{len(keywords)}",
            'title': title,
            'content_type': content_type,
            'primary_keywords': primary_keywords,
            'secondary_keywords': secondary_keywords,
            'seo_optimization_score': seo_score,
            'traffic_potential_score': traffic_score,
            'combined_score': combined_score,
            'total_search_volume': total_search_volume,
            'average_difficulty': round(avg_difficulty, 2),
            'average_cpc': round(avg_cpc, 2),
            'optimization_tips': optimization_tips,
            'content_outline': content_outline
        }
    
    def _generate_content_title(self, topic: str, primary_keywords: List[str]) -> str:
        """Generate content title from topic and keywords"""
        
        topic_titles = {
            'project_management': 'Project Management',
            'software_tools': 'Software Tools',
            'marketing': 'Marketing',
            'business': 'Business',
            'technology': 'Technology',
            'productivity': 'Productivity'
        }
        
        base_topic = topic_titles.get(topic, 'Tools')
        
        # Use first primary keyword for title
        if primary_keywords:
            first_keyword = primary_keywords[0]
            return f"Best {base_topic} {first_keyword} in 2024"
        
        return f"Best {base_topic} Tools in 2024"
    
    def _determine_content_type(self, primary_keywords: List[str]) -> str:
        """Determine content type from primary keywords"""
        
        keyword_text = ' '.join(primary_keywords).lower()
        
        if any(pattern in keyword_text for pattern in ['best', 'top', 'list']):
            return 'list-article'
        elif any(pattern in keyword_text for pattern in ['how to', 'guide', 'tutorial']):
            return 'how-to-guide'
        elif any(pattern in keyword_text for pattern in ['vs', 'versus', 'compare']):
            return 'comparison'
        elif any(pattern in keyword_text for pattern in ['review', 'rating']):
            return 'review'
        else:
            return 'list-article'  # Default
    
    def _calculate_seo_score(self, keywords: List[Dict[str, Any]]) -> float:
        """Calculate SEO optimization score"""
        
        if not keywords:
            return 0.0
        
        # Factors for SEO score
        avg_opportunity = sum(k.get('opportunity_score', 0) for k in keywords) / len(keywords)
        keyword_diversity = len(set(k.get('Keyword', '') for k in keywords))
        intent_variety = len(set(k.get('primary_intent', '') for k in keywords))
        
        # Calculate score (0-100)
        seo_score = (avg_opportunity * 0.6 + keyword_diversity * 0.2 + intent_variety * 0.2)
        
        return min(100, max(0, seo_score))
    
    def _calculate_traffic_score(self, keywords: List[Dict[str, Any]]) -> float:
        """Calculate traffic potential score"""
        
        if not keywords:
            return 0.0
        
        # Factors for traffic score
        total_volume = sum(k.get('Volume', 0) for k in keywords)
        avg_difficulty = sum(k.get('Difficulty', 0) for k in keywords) / len(keywords)
        
        # Calculate score (0-100)
        volume_score = min(100, total_volume / 1000)  # Normalize volume
        difficulty_score = max(0, 100 - avg_difficulty)  # Invert difficulty
        
        traffic_score = (volume_score * 0.7 + difficulty_score * 0.3)
        
        return min(100, max(0, traffic_score))
    
    def _generate_optimization_tips(self, primary_keywords: List[str], secondary_keywords: List[str]) -> List[str]:
        """Generate optimization tips for content"""
        
        tips = []
        
        if primary_keywords:
            first_keyword = primary_keywords[0]
            tips.append(f"Include '{first_keyword}' in your title and first paragraph")
        
        if len(primary_keywords) > 1:
            tips.append(f"Use other primary keywords like '{primary_keywords[1]}' in H2 headings")
        
        if secondary_keywords:
            tips.append(f"Include secondary keywords like '{secondary_keywords[0]}' in H3 headings")
        
        tips.append("Add internal links to related content")
        tips.append("Include meta description with target keywords")
        
        return tips
    
    def _generate_content_outline(self, content_type: str, primary_keywords: List[str]) -> str:
        """Generate content outline based on type"""
        
        outlines = {
            'list-article': 'Introduction → Top 10 Items → Detailed Reviews → Comparison Table → Conclusion',
            'how-to-guide': 'Introduction → Step-by-Step Guide → Tips & Best Practices → Conclusion',
            'comparison': 'Introduction → Feature Comparison → Pros & Cons → Recommendation → Conclusion',
            'review': 'Introduction → Overview → Features → Pros & Cons → Rating → Conclusion'
        }
        
        return outlines.get(content_type, 'Introduction → Main Content → Conclusion')
    
    async def save_report(self, report: Dict[str, Any]) -> str:
        """Save report to database"""
        try:
            report_id = report.get('report_id', 'unknown')
            
            # Save report to database
            success = await self.db_service.save_report(report)
            
            if success:
                logger.info(f"Saved report {report_id}")
                return report_id
            else:
                logger.error(f"Failed to save report {report_id}")
                raise ValueError(f"Failed to save report {report_id}")
                
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise ValueError(f"Error saving report: {str(e)}")
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report from database"""
        try:
            return await self.db_service.get_report(report_id)
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}")
            return None
    
    async def get_user_reports(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all reports for a user"""
        try:
            return await self.db_service.get_user_reports(user_id)
        except Exception as e:
            logger.error(f"Error getting user reports: {str(e)}")
            return []
    
    async def export_report_to_csv(self, report: Dict[str, Any]) -> str:
        """Export report to CSV format"""
        try:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write summary
            writer.writerow(['Section', 'Metric', 'Value'])
            summary = report.get('summary', {})
            for key, value in summary.items():
                writer.writerow(['Summary', key, value])
            
            # Write top opportunities
            writer.writerow([])
            writer.writerow(['Top Opportunities'])
            top_opportunities = report.get('top_opportunities', {})
            for category, keywords in top_opportunities.items():
                writer.writerow([category])
                for keyword in keywords:
                    writer.writerow(['', keyword.get('Keyword', ''), keyword.get('opportunity_score', 0)])
            
            # Write content recommendations
            writer.writerow([])
            writer.writerow(['Content Recommendations'])
            recommendations = report.get('content_recommendations', [])
            for rec in recommendations:
                writer.writerow([rec.get('keyword', ''), rec.get('content_format', ''), rec.get('seo_score', 0)])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting report to CSV: {str(e)}")
            raise ValueError(f"Error exporting report to CSV: {str(e)}")
    
    async def export_report_to_json(self, report: Dict[str, Any]) -> str:
        """Export report to JSON format"""
        try:
            return json.dumps(report, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting report to JSON: {str(e)}")
            raise ValueError(f"Error exporting report to JSON: {str(e)}")

# Global instance
report_generator_service = ReportGeneratorService()

