"""
Enhanced database service using Supabase SDK for Ahrefs integration
"""

import logging
from typing import Dict, List, Any, Optional
from supabase import Client
from ..core.supabase_singleton import get_supabase_client

logger = logging.getLogger(__name__)

class EnhancedDatabaseService:
    """Enhanced database service using Supabase SDK for Ahrefs integration"""
    
    def __init__(self):
        self.supabase: Client = get_supabase_client()
    
    async def save_ahrefs_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Save Ahrefs analysis data to Supabase"""
        try:
            # Save to ahrefs_analyses table
            result = self.supabase.table('ahrefs_analyses').insert({
                'user_id': analysis_data['user_id'],
                'file_id': analysis_data['file_id'],
                'analysis_id': analysis_data['analysis_id'],
                'summary': analysis_data['summary'],
                'top_opportunities': analysis_data['top_opportunities'],
                'content_recommendations': analysis_data['content_recommendations'],
                'insights': analysis_data['insights'],
                'next_steps': analysis_data['next_steps'],
                'seo_content_ideas': analysis_data['seo_content_ideas'],
                'status': 'completed',
                'created_at': 'now()'
            }).execute()
            
            if result.data:
                analysis_id = result.data[0]['id']
                logger.info(f"Saved Ahrefs analysis {analysis_id}")
                return analysis_id
            else:
                raise ValueError("Failed to save Ahrefs analysis")
                
        except Exception as e:
            logger.error(f"Error saving Ahrefs analysis: {str(e)}")
            raise ValueError(f"Failed to save Ahrefs analysis: {str(e)}")
    
    async def get_ahrefs_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get Ahrefs analysis by ID"""
        try:
            result = self.supabase.table('ahrefs_analyses').select('*').eq('id', analysis_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting Ahrefs analysis {analysis_id}: {str(e)}")
            return None
    
    async def save_enhanced_ideas(self, ideas_data: Dict[str, Any]) -> List[str]:
        """Save enhanced ideas (blog and software) to Supabase"""
        try:
            idea_ids = []
            
            # Save blog ideas
            if ideas_data.get('blog_ideas'):
                blog_result = self.supabase.table('blog_ideas').insert([
                    {
                        'id': idea['id'],
                        'user_id': ideas_data['user_id'],
                        'analysis_id': ideas_data['analysis_id'],
                        'title': idea['title'],
                        'content_type': idea['content_type'],
                        'primary_keywords': idea['primary_keywords'],
                        'secondary_keywords': idea['secondary_keywords'],
                        'seo_optimization_score': idea['seo_optimization_score'],
                        'traffic_potential_score': idea['traffic_potential_score'],
                        'combined_score': idea['combined_score'],
                        'total_search_volume': idea['total_search_volume'],
                        'average_difficulty': idea['average_difficulty'],
                        'average_cpc': idea['average_cpc'],
                        'optimization_tips': idea['optimization_tips'],
                        'content_outline': idea['content_outline'],
                        'target_audience': idea['target_audience'],
                        'content_length': idea['content_length'],
                        'enhanced_with_ahrefs': idea.get('enhanced_with_ahrefs', False),
                        'created_at': 'now()'
                    }
                    for idea in ideas_data['blog_ideas']
                ]).execute()
                
                if blog_result.data:
                    idea_ids.extend([idea['id'] for idea in blog_result.data])
            
            # Save software ideas
            if ideas_data.get('software_ideas'):
                software_result = self.supabase.table('software_ideas').insert([
                    {
                        'id': idea['id'],
                        'user_id': ideas_data['user_id'],
                        'analysis_id': ideas_data['analysis_id'],
                        'title': idea['title'],
                        'description': idea['description'],
                        'features': idea['features'],
                        'target_market': idea['target_market'],
                        'monetization_strategy': idea['monetization_strategy'],
                        'technical_requirements': idea['technical_requirements'],
                        'market_opportunity_score': idea['market_opportunity_score'],
                        'development_difficulty': idea['development_difficulty'],
                        'estimated_development_time': idea['estimated_development_time'],
                        'enhanced_with_ahrefs': idea.get('enhanced_with_ahrefs', False),
                        'created_at': 'now()'
                    }
                    for idea in ideas_data['software_ideas']
                ]).execute()
                
                if software_result.data:
                    idea_ids.extend([idea['id'] for idea in software_result.data])
            
            logger.info(f"Saved {len(idea_ids)} enhanced ideas")
            return idea_ids
            
        except Exception as e:
            logger.error(f"Error saving enhanced ideas: {str(e)}")
            raise ValueError(f"Failed to save enhanced ideas: {str(e)}")
    
    async def get_enhanced_ideas(self, analysis_id: str, idea_type: str = 'all') -> Dict[str, List[Dict[str, Any]]]:
        """Get enhanced ideas by analysis ID"""
        try:
            result = {'blog_ideas': [], 'software_ideas': []}
            
            if idea_type in ['all', 'blog']:
                blog_result = self.supabase.table('blog_ideas').select('*').eq('analysis_id', analysis_id).execute()
                if blog_result.data:
                    result['blog_ideas'] = blog_result.data
            
            if idea_type in ['all', 'software']:
                software_result = self.supabase.table('software_ideas').select('*').eq('analysis_id', analysis_id).execute()
                if software_result.data:
                    result['software_ideas'] = software_result.data
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting enhanced ideas: {str(e)}")
            return {'blog_ideas': [], 'software_ideas': []}
    
    async def save_ahrefs_file(self, file_data: Dict[str, Any]) -> str:
        """Save Ahrefs file metadata to Supabase"""
        try:
            result = self.supabase.table('ahrefs_files').insert({
                'user_id': file_data['user_id'],
                'filename': file_data['filename'],
                'file_size': file_data['file_size'],
                'file_path': file_data['file_path'],
                'status': file_data['status'],
                'uploaded_at': 'now()'
            }).execute()
            
            if result.data:
                file_id = result.data[0]['id']
                logger.info(f"Saved Ahrefs file {file_id}")
                return file_id
            else:
                raise ValueError("Failed to save Ahrefs file")
                
        except Exception as e:
            logger.error(f"Error saving Ahrefs file: {str(e)}")
            raise ValueError(f"Failed to save Ahrefs file: {str(e)}")
    
    async def get_ahrefs_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get Ahrefs file by ID"""
        try:
            result = self.supabase.table('ahrefs_files').select('*').eq('id', file_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting Ahrefs file {file_id}: {str(e)}")
            return None
    
    async def update_ahrefs_file_status(self, file_id: str, status: str, error_message: str = None):
        """Update Ahrefs file status"""
        try:
            update_data = {'status': status, 'updated_at': 'now()'}
            if error_message:
                update_data['error_message'] = error_message
            
            result = self.supabase.table('ahrefs_files').update(update_data).eq('id', file_id).execute()
            
            if result.data:
                logger.info(f"Updated Ahrefs file {file_id} status to {status}")
            else:
                logger.warning(f"Failed to update Ahrefs file {file_id} status")
                
        except Exception as e:
            logger.error(f"Error updating Ahrefs file status: {str(e)}")
            raise ValueError(f"Failed to update Ahrefs file status: {str(e)}")
    
    async def save_keyword_metrics(self, keywords_data: List[Dict[str, Any]]) -> List[str]:
        """Save keyword metrics to Supabase"""
        try:
            result = self.supabase.table('keyword_metrics').insert([
                {
                    'keyword': keyword['keyword'],
                    'search_volume': keyword['search_volume'],
                    'difficulty': keyword['difficulty'],
                    'cpc': keyword['cpc'],
                    'intents': keyword['intents'],
                    'opportunity_score': keyword['opportunity_score'],
                    'category': keyword['category'],
                    'primary_intent': keyword['primary_intent'],
                    'created_at': 'now()'
                }
                for keyword in keywords_data
            ]).execute()
            
            if result.data:
                keyword_ids = [keyword['id'] for keyword in result.data]
                logger.info(f"Saved {len(keyword_ids)} keyword metrics")
                return keyword_ids
            else:
                raise ValueError("Failed to save keyword metrics")
                
        except Exception as e:
            logger.error(f"Error saving keyword metrics: {str(e)}")
            raise ValueError(f"Failed to save keyword metrics: {str(e)}")
    
    async def get_keyword_metrics(self, analysis_id: str) -> List[Dict[str, Any]]:
        """Get keyword metrics by analysis ID"""
        try:
            result = self.supabase.table('keyword_metrics').select('*').eq('analysis_id', analysis_id).execute()
            
            if result.data:
                return result.data
            return []
            
        except Exception as e:
            logger.error(f"Error getting keyword metrics: {str(e)}")
            return []
    
    async def get_user_analyses(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's Ahrefs analyses"""
        try:
            result = self.supabase.table('ahrefs_analyses').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            return []
            
        except Exception as e:
            logger.error(f"Error getting user analyses: {str(e)}")
            return []
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis and related data"""
        try:
            # Delete related ideas first
            self.supabase.table('blog_ideas').delete().eq('analysis_id', analysis_id).execute()
            self.supabase.table('software_ideas').delete().eq('analysis_id', analysis_id).execute()
            self.supabase.table('keyword_metrics').delete().eq('analysis_id', analysis_id).execute()
            
            # Delete analysis
            result = self.supabase.table('ahrefs_analyses').delete().eq('id', analysis_id).execute()
            
            if result.data:
                logger.info(f"Deleted analysis {analysis_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting analysis: {str(e)}")
            return False
    
    async def get_analysis_stats(self, user_id: str) -> Dict[str, Any]:
        """Get analysis statistics for user"""
        try:
            # Get total analyses
            analyses_result = self.supabase.table('ahrefs_analyses').select('id').eq('user_id', user_id).execute()
            total_analyses = len(analyses_result.data) if analyses_result.data else 0
            
            # Get total ideas
            blog_result = self.supabase.table('blog_ideas').select('id').eq('user_id', user_id).execute()
            software_result = self.supabase.table('software_ideas').select('id').eq('user_id', user_id).execute()
            
            total_blog_ideas = len(blog_result.data) if blog_result.data else 0
            total_software_ideas = len(software_result.data) if software_result.data else 0
            
            return {
                'total_analyses': total_analyses,
                'total_blog_ideas': total_blog_ideas,
                'total_software_ideas': total_software_ideas,
                'total_ideas': total_blog_ideas + total_software_ideas
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis stats: {str(e)}")
            return {
                'total_analyses': 0,
                'total_blog_ideas': 0,
                'total_software_ideas': 0,
                'total_ideas': 0
            }
    
    async def search_ideas(self, user_id: str, query: str, idea_type: str = 'all') -> Dict[str, List[Dict[str, Any]]]:
        """Search ideas by query"""
        try:
            result = {'blog_ideas': [], 'software_ideas': []}
            
            if idea_type in ['all', 'blog']:
                blog_result = self.supabase.table('blog_ideas').select('*').eq('user_id', user_id).ilike('title', f'%{query}%').execute()
                if blog_result.data:
                    result['blog_ideas'] = blog_result.data
            
            if idea_type in ['all', 'software']:
                software_result = self.supabase.table('software_ideas').select('*').eq('user_id', user_id).ilike('title', f'%{query}%').execute()
                if software_result.data:
                    result['software_ideas'] = software_result.data
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching ideas: {str(e)}")
            return {'blog_ideas': [], 'software_ideas': []}
    
    async def get_idea_by_id(self, idea_id: str, idea_type: str) -> Optional[Dict[str, Any]]:
        """Get idea by ID and type"""
        try:
            table_name = 'blog_ideas' if idea_type == 'blog' else 'software_ideas'
            result = self.supabase.table(table_name).select('*').eq('id', idea_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting idea {idea_id}: {str(e)}")
            return None
    
    async def update_idea(self, idea_id: str, idea_type: str, updates: Dict[str, Any]) -> bool:
        """Update idea"""
        try:
            table_name = 'blog_ideas' if idea_type == 'blog' else 'software_ideas'
            result = self.supabase.table(table_name).update(updates).eq('id', idea_id).execute()
            
            if result.data:
                logger.info(f"Updated {idea_type} idea {idea_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating idea {idea_id}: {str(e)}")
            return False
    
    async def delete_idea(self, idea_id: str, idea_type: str) -> bool:
        """Delete idea"""
        try:
            table_name = 'blog_ideas' if idea_type == 'blog' else 'software_ideas'
            result = self.supabase.table(table_name).delete().eq('id', idea_id).execute()
            
            if result.data:
                logger.info(f"Deleted {idea_type} idea {idea_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting idea {idea_id}: {str(e)}")
            return False

# Global instance
enhanced_database_service = EnhancedDatabaseService()

