"""
Keyword Clustering Service
Handles keyword clustering using scikit-learn and external tool data
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ..models.keyword_cluster import KeywordCluster
from ..models.external_tool_result import ExternalToolResult
from ..core.redis import cache

logger = structlog.get_logger()

class KeywordClusteringService:
    """Service for keyword clustering and analysis"""
    
    def __init__(self, db: Session):
        self.db = db
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def cluster_keywords(
        self,
        user_id: str,
        workflow_session_id: str,
        external_tool_result_id: Optional[str] = None,
        keywords_data: Optional[List[Dict[str, Any]]] = None,
        cluster_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 3,
        max_clusters: int = 20
    ) -> Dict[str, Any]:
        """Cluster keywords using specified method"""
        try:
            logger.info("Starting keyword clustering", 
                       user_id=user_id, 
                       workflow_session_id=workflow_session_id,
                       cluster_method=cluster_method)
            
            # Get keywords data
            if external_tool_result_id:
                keywords = await self._get_keywords_from_external_tool(external_tool_result_id, user_id)
            elif keywords_data:
                keywords = keywords_data
            else:
                raise ValueError("Either external_tool_result_id or keywords_data must be provided")
            
            if not keywords:
                raise ValueError("No keywords found for clustering")
            
            # Prepare keywords for clustering
            prepared_keywords = self._prepare_keywords_for_clustering(keywords)
            
            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = self._determine_optimal_clusters(prepared_keywords, max_clusters)
            
            # Perform clustering
            clusters = await self._perform_clustering(
                prepared_keywords, 
                cluster_method, 
                n_clusters, 
                min_cluster_size
            )
            
            # Create cluster objects
            cluster_objects = await self._create_cluster_objects(
                user_id=user_id,
                workflow_session_id=workflow_session_id,
                external_tool_result_id=external_tool_result_id,
                clusters=clusters,
                original_keywords=keywords
            )
            
            # Update external tool result if provided
            if external_tool_result_id:
                await self._update_external_tool_result(external_tool_result_id, cluster_objects, user_id)
            
            logger.info("Keyword clustering completed successfully", 
                       clusters_created=len(cluster_objects),
                       user_id=user_id)
            
            return {
                "clusters": [cluster.to_dict() for cluster in cluster_objects],
                "clustering_metadata": {
                    "method": cluster_method,
                    "n_clusters": n_clusters,
                    "total_keywords": len(keywords),
                    "clustered_keywords": sum(len(cluster.keywords) for cluster in cluster_objects),
                    "clustered_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to cluster keywords", 
                        error=str(e), 
                        user_id=user_id)
            raise
    
    async def _get_keywords_from_external_tool(self, external_tool_result_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get keywords from external tool result"""
        try:
            result = self.db.query(ExternalToolResult).filter(
                ExternalToolResult.id == external_tool_result_id,
                ExternalToolResult.user_id == user_id
            ).first()
            
            if not result:
                raise ValueError("External tool result not found")
            
            return result.keywords_data or []
            
        except Exception as e:
            logger.error("Failed to get keywords from external tool", error=str(e))
            raise
    
    def _prepare_keywords_for_clustering(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare keywords for clustering"""
        prepared = []
        
        for keyword in keywords:
            if isinstance(keyword, dict):
                # Extract keyword text
                keyword_text = keyword.get("keyword", "")
                if not keyword_text:
                    continue
                
                # Prepare features for clustering
                prepared_keyword = {
                    "original": keyword,
                    "text": keyword_text.lower().strip(),
                    "search_volume": keyword.get("search_volume", 0),
                    "difficulty": keyword.get("difficulty", 0),
                    "cpc": keyword.get("cpc", 0),
                    "intent": self._determine_search_intent(keyword_text),
                    "length": len(keyword_text.split()),
                    "features": self._extract_keyword_features(keyword_text)
                }
                
                prepared.append(prepared_keyword)
        
        return prepared
    
    def _determine_search_intent(self, keyword: str) -> str:
        """Determine search intent from keyword"""
        keyword_lower = keyword.lower()
        
        # Informational intent
        if any(word in keyword_lower for word in ['what', 'how', 'why', 'when', 'where', 'guide', 'tutorial', 'learn']):
            return "informational"
        
        # Navigational intent
        elif any(word in keyword_lower for word in ['login', 'sign in', 'website', 'official']):
            return "navigational"
        
        # Transactional intent
        elif any(word in keyword_lower for word in ['buy', 'purchase', 'order', 'price', 'cost', 'deal', 'sale']):
            return "transactional"
        
        # Commercial investigation
        elif any(word in keyword_lower for word in ['best', 'review', 'compare', 'vs', 'alternative']):
            return "commercial"
        
        else:
            return "informational"  # Default
    
    def _extract_keyword_features(self, keyword: str) -> List[str]:
        """Extract features from keyword for clustering"""
        features = []
        
        # Word count
        word_count = len(keyword.split())
        features.append(f"words_{word_count}")
        
        # Length category
        if len(keyword) <= 20:
            features.append("short")
        elif len(keyword) <= 50:
            features.append("medium")
        else:
            features.append("long")
        
        # Question words
        if any(word in keyword.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            features.append("question")
        
        # Action words
        if any(word in keyword.lower() for word in ['buy', 'get', 'find', 'learn', 'make', 'create']):
            features.append("action")
        
        # Brand words
        if any(word in keyword.lower() for word in ['brand', 'company', 'official', 'app', 'software']):
            features.append("brand")
        
        # Location words
        if any(word in keyword.lower() for word in ['near', 'local', 'city', 'state', 'country']):
            features.append("location")
        
        return features
    
    def _determine_optimal_clusters(self, keywords: List[Dict[str, Any]], max_clusters: int) -> int:
        """Determine optimal number of clusters using elbow method"""
        if len(keywords) < 4:
            return 2
        
        # Use TF-IDF for text similarity
        texts = [kw["text"] for kw in keywords]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate inertia for different cluster numbers
        inertias = []
        max_k = min(max_clusters, len(keywords) // 2, 10)
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 3:
            return 2
        
        # Simple elbow detection
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        if diffs:
            elbow_idx = diffs.index(max(diffs)) + 2
            return min(elbow_idx, max_clusters)
        
        return 3  # Default
    
    async def _perform_clustering(
        self,
        keywords: List[Dict[str, Any]],
        method: str,
        n_clusters: int,
        min_cluster_size: int
    ) -> List[Dict[str, Any]]:
        """Perform clustering using specified method"""
        try:
            # Prepare text data
            texts = [kw["text"] for kw in keywords]
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform clustering
            if method == "kmeans":
                clusters = self._kmeans_clustering(tfidf_matrix, n_clusters)
            elif method == "dbscan":
                clusters = self._dbscan_clustering(tfidf_matrix, min_cluster_size)
            elif method == "agglomerative":
                clusters = self._agglomerative_clustering(tfidf_matrix, n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Group keywords by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(keywords[i])
            
            # Filter out small clusters
            filtered_clusters = {
                cid: keywords for cid, keywords in cluster_groups.items()
                if len(keywords) >= min_cluster_size
            }
            
            return list(filtered_clusters.values())
            
        except Exception as e:
            logger.error("Failed to perform clustering", error=str(e))
            raise
    
    def _kmeans_clustering(self, tfidf_matrix, n_clusters: int) -> List[int]:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(tfidf_matrix).tolist()
    
    def _dbscan_clustering(self, tfidf_matrix, min_cluster_size: int) -> List[int]:
        """Perform DBSCAN clustering"""
        dbscan = DBSCAN(eps=0.3, min_samples=min_cluster_size)
        return dbscan.fit_predict(tfidf_matrix).tolist()
    
    def _agglomerative_clustering(self, tfidf_matrix, n_clusters: int) -> List[int]:
        """Perform Agglomerative clustering"""
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return agg_clustering.fit_predict(tfidf_matrix.toarray()).tolist()
    
    async def _create_cluster_objects(
        self,
        user_id: str,
        workflow_session_id: str,
        external_tool_result_id: Optional[str],
        clusters: List[List[Dict[str, Any]]],
        original_keywords: List[Dict[str, Any]]
    ) -> List[KeywordCluster]:
        """Create KeywordCluster objects from clustered data"""
        cluster_objects = []
        
        for i, cluster_keywords in enumerate(clusters):
            try:
                # Calculate cluster metrics
                metrics = self._calculate_cluster_metrics(cluster_keywords)
                
                # Determine primary keyword
                primary_keyword = self._determine_primary_keyword(cluster_keywords)
                
                # Extract secondary and long-tail keywords
                secondary_keywords = self._extract_secondary_keywords(cluster_keywords)
                long_tail_keywords = self._extract_long_tail_keywords(cluster_keywords)
                
                # Generate content ideas and angles
                content_ideas = self._generate_content_ideas_for_cluster(cluster_keywords)
                content_angles = self._generate_content_angles_for_cluster(cluster_keywords)
                target_audiences = self._generate_target_audiences_for_cluster(cluster_keywords)
                
                # Create cluster object
                cluster = KeywordCluster(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    workflow_session_id=workflow_session_id,
                    trend_analysis_id=None,  # Will be set if needed
                    cluster_name=f"Cluster {i+1}: {primary_keyword}",
                    cluster_description=f"Keyword cluster focused on {primary_keyword} and related terms",
                    cluster_type="semantic",
                    keywords=[kw["original"] for kw in cluster_keywords],
                    primary_keyword=primary_keyword,
                    secondary_keywords=secondary_keywords,
                    long_tail_keywords=long_tail_keywords,
                    avg_search_volume=metrics["avg_search_volume"],
                    avg_keyword_difficulty=metrics["avg_difficulty"],
                    avg_cpc=metrics["avg_cpc"],
                    total_search_volume=metrics["total_search_volume"],
                    competition_level=metrics["competition_level"],
                    cluster_size=len(cluster_keywords),
                    cluster_density=metrics["cluster_density"],
                    semantic_similarity=metrics["semantic_similarity"],
                    intent_consistency=metrics["intent_consistency"],
                    content_ideas=content_ideas,
                    content_angles=content_angles,
                    target_audiences=target_audiences,
                    source_tool="clustering_service",
                    external_data={},
                    cluster_quality_score=metrics["quality_score"],
                    keyword_relevance_score=metrics["relevance_score"],
                    content_potential_score=metrics["content_potential"],
                    is_active=True,
                    is_processed=True
                )
                
                self.db.add(cluster)
                cluster_objects.append(cluster)
                
            except Exception as e:
                logger.error(f"Failed to create cluster object {i}", error=str(e))
                continue
        
        self.db.commit()
        
        # Refresh all objects
        for cluster in cluster_objects:
            self.db.refresh(cluster)
        
        return cluster_objects
    
    def _calculate_cluster_metrics(self, cluster_keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a cluster"""
        if not cluster_keywords:
            return {}
        
        # Extract numerical data
        volumes = [kw["search_volume"] for kw in cluster_keywords]
        difficulties = [kw["difficulty"] for kw in cluster_keywords]
        cpcs = [kw["cpc"] for kw in cluster_keywords]
        
        # Calculate averages
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        avg_cpc = sum(cpcs) / len(cpcs) if cpcs else 0
        total_volume = sum(volumes)
        
        # Determine competition level
        if avg_difficulty <= 30:
            competition_level = "low"
        elif avg_difficulty <= 60:
            competition_level = "medium"
        else:
            competition_level = "high"
        
        # Calculate cluster density (simplified)
        cluster_density = min(1.0, len(cluster_keywords) / 10.0)
        
        # Calculate semantic similarity (placeholder)
        semantic_similarity = 0.8  # Would need actual NLP analysis
        
        # Calculate intent consistency
        intents = [kw["intent"] for kw in cluster_keywords]
        intent_consistency = len(set(intents)) / len(intents) if intents else 0
        
        # Calculate quality scores
        quality_score = (cluster_density * 0.3 + semantic_similarity * 0.3 + 
                        intent_consistency * 0.2 + (1.0 - avg_difficulty/100) * 0.2) * 100
        
        relevance_score = min(100, avg_volume / 100)  # Simple relevance based on volume
        
        content_potential = (relevance_score * 0.4 + (100 - avg_difficulty) * 0.3 + 
                           cluster_density * 100 * 0.3)
        
        return {
            "avg_search_volume": int(avg_volume),
            "avg_difficulty": round(avg_difficulty, 2),
            "avg_cpc": round(avg_cpc, 2),
            "total_search_volume": int(total_volume),
            "competition_level": competition_level,
            "cluster_density": round(cluster_density, 3),
            "semantic_similarity": round(semantic_similarity, 3),
            "intent_consistency": round(intent_consistency, 3),
            "quality_score": round(quality_score, 2),
            "relevance_score": round(relevance_score, 2),
            "content_potential": round(content_potential, 2)
        }
    
    def _determine_primary_keyword(self, cluster_keywords: List[Dict[str, Any]]) -> str:
        """Determine primary keyword for cluster"""
        if not cluster_keywords:
            return "unknown"
        
        # Sort by search volume and difficulty
        sorted_keywords = sorted(
            cluster_keywords,
            key=lambda x: (x["search_volume"], -x["difficulty"]),
            reverse=True
        )
        
        return sorted_keywords[0]["text"]
    
    def _extract_secondary_keywords(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Extract secondary keywords from cluster"""
        if len(cluster_keywords) <= 1:
            return []
        
        # Get top keywords by volume, excluding primary
        sorted_keywords = sorted(
            cluster_keywords,
            key=lambda x: x["search_volume"],
            reverse=True
        )
        
        return [kw["text"] for kw in sorted_keywords[1:6]]  # Top 5 secondary keywords
    
    def _extract_long_tail_keywords(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Extract long-tail keywords from cluster"""
        long_tail = []
        
        for kw in cluster_keywords:
            if kw["length"] >= 3:  # 3+ words
                long_tail.append(kw["text"])
        
        return long_tail[:10]  # Limit to 10 long-tail keywords
    
    def _generate_content_ideas_for_cluster(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate content ideas for cluster"""
        primary = self._determine_primary_keyword(cluster_keywords)
        
        ideas = [
            f"Complete Guide to {primary.title()}",
            f"Best {primary.title()} Tips and Tricks",
            f"{primary.title()} vs Alternatives: Comparison",
            f"How to Choose the Right {primary.title()}",
            f"{primary.title()} for Beginners"
        ]
        
        return ideas[:5]
    
    def _generate_content_angles_for_cluster(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate content angles for cluster"""
        angles = [
            "Comprehensive guide",
            "Beginner-friendly tutorial",
            "Expert comparison",
            "Cost-benefit analysis",
            "Step-by-step instructions"
        ]
        
        return angles[:5]
    
    def _generate_target_audiences_for_cluster(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate target audiences for cluster"""
        audiences = [
            "Beginners",
            "Professionals",
            "Enthusiasts",
            "Business owners",
            "Students"
        ]
        
        return audiences[:5]
    
    async def _update_external_tool_result(
        self, 
        external_tool_result_id: str, 
        clusters: List[KeywordCluster], 
        user_id: str
    ) -> None:
        """Update external tool result with cluster data"""
        try:
            result = self.db.query(ExternalToolResult).filter(
                ExternalToolResult.id == external_tool_result_id,
                ExternalToolResult.user_id == user_id
            ).first()
            
            if result:
                result.clusters_data = [cluster.to_dict() for cluster in clusters]
                result.total_clusters = len(clusters)
                result.is_processed = True
                result.processed_at = datetime.utcnow()
                
                self.db.commit()
                
        except Exception as e:
            logger.error("Failed to update external tool result", error=str(e))
            self.db.rollback()
    
    async def get_clusters(
        self,
        user_id: str,
        workflow_session_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_processed: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get keyword clusters with optional filtering"""
        try:
            query = self.db.query(KeywordCluster).filter(KeywordCluster.user_id == user_id)
            
            if workflow_session_id:
                query = query.filter(KeywordCluster.workflow_session_id == workflow_session_id)
            
            if is_active is not None:
                query = query.filter(KeywordCluster.is_active == is_active)
            
            if is_processed is not None:
                query = query.filter(KeywordCluster.is_processed == is_processed)
            
            clusters = query.order_by(KeywordCluster.created_at.desc()).offset(offset).limit(limit).all()
            
            return [cluster.to_dict() for cluster in clusters]
            
        except Exception as e:
            logger.error("Failed to get clusters", error=str(e))
            raise
    
    async def update_cluster(
        self,
        cluster_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[KeywordCluster]:
        """Update keyword cluster"""
        try:
            cluster = self.db.query(KeywordCluster).filter(
                KeywordCluster.id == cluster_id,
                KeywordCluster.user_id == user_id
            ).first()
            
            if not cluster:
                return None
            
            # Update allowed fields
            allowed_fields = [
                "cluster_name", "cluster_description", "content_ideas", 
                "content_angles", "target_audiences", "processing_notes", 
                "is_active", "is_used_for_content"
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(cluster, field):
                    setattr(cluster, field, value)
            
            cluster.updated_at = datetime.utcnow()
            self.db.commit()
            
            return cluster
            
        except Exception as e:
            logger.error("Failed to update cluster", error=str(e))
            self.db.rollback()
            raise
    
    async def delete_cluster(self, cluster_id: str, user_id: str) -> bool:
        """Delete keyword cluster"""
        try:
            cluster = self.db.query(KeywordCluster).filter(
                KeywordCluster.id == cluster_id,
                KeywordCluster.user_id == user_id
            ).first()
            
            if not cluster:
                return False
            
            self.db.delete(cluster)
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete cluster", error=str(e))
            self.db.rollback()
            raise