"""
Keyword Clustering Utility

Groups related keywords using semantic similarity and topic modeling.
Uses scikit-learn for clustering algorithms.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from collections import Counter

from ..models.keyword import Keyword

logger = logging.getLogger(__name__)

class KeywordClustering:
    """Utility for clustering related keywords"""
    
    def __init__(self, min_cluster_size: int = 3, max_clusters: int = 10):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def cluster_keywords(self, keywords: List[Keyword]) -> List[List[str]]:
        """
        Cluster keywords by semantic similarity
        
        Args:
            keywords: List of keywords to cluster
            
        Returns:
            List of keyword clusters (each cluster is a list of keyword strings)
        """
        try:
            if len(keywords) < self.min_cluster_size:
                logger.warning(f"Not enough keywords for clustering: {len(keywords)}")
                return [[k.keyword for k in keywords]]
            
            # Extract keyword texts
            keyword_texts = [k.keyword for k in keywords]
            
            # Preprocess keywords
            processed_keywords = self._preprocess_keywords(keyword_texts)
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(processed_keywords)
            
            # Determine optimal number of clusters
            optimal_k = self._find_optimal_clusters(tfidf_matrix, len(keywords))
            
            # Perform clustering
            if optimal_k <= 1:
                return [keyword_texts]
            
            clusters = self._perform_clustering(tfidf_matrix, optimal_k)
            
            # Group keywords by cluster
            keyword_clusters = self._group_keywords_by_cluster(keyword_texts, clusters)
            
            # Filter out small clusters
            filtered_clusters = [
                cluster for cluster in keyword_clusters 
                if len(cluster) >= self.min_cluster_size
            ]
            
            logger.info(f"Created {len(filtered_clusters)} keyword clusters")
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Error clustering keywords: {str(e)}")
            # Return single cluster with all keywords as fallback
            return [[k.keyword for k in keywords]]
    
    def _preprocess_keywords(self, keywords: List[str]) -> List[str]:
        """Preprocess keywords for better clustering"""
        processed = []
        
        for keyword in keywords:
            # Convert to lowercase
            processed_keyword = keyword.lower()
            
            # Remove special characters but keep spaces
            processed_keyword = re.sub(r'[^\w\s]', ' ', processed_keyword)
            
            # Remove extra whitespace
            processed_keyword = ' '.join(processed_keyword.split())
            
            processed.append(processed_keyword)
        
        return processed
    
    def _find_optimal_clusters(self, tfidf_matrix, num_keywords: int) -> int:
        """Find optimal number of clusters using elbow method"""
        if num_keywords <= 3:
            return 1
        
        max_k = min(self.max_clusters, num_keywords // 2)
        if max_k < 2:
            return 1
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 2:
            return 2
        
        # Calculate second derivative to find elbow
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        if second_derivatives:
            elbow_index = second_derivatives.index(max(second_derivatives)) + 2
            return k_range[elbow_index - 2]
        
        return 2
    
    def _perform_clustering(self, tfidf_matrix, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        return cluster_labels
    
    def _group_keywords_by_cluster(
        self, 
        keywords: List[str], 
        cluster_labels: np.ndarray
    ) -> List[List[str]]:
        """Group keywords by their cluster assignments"""
        clusters = {}
        
        for i, keyword in enumerate(keywords):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(keyword)
        
        return list(clusters.values())
    
    def find_similar_keywords(
        self, 
        target_keyword: str, 
        all_keywords: List[Keyword], 
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find keywords similar to a target keyword
        
        Args:
            target_keyword: The keyword to find similarities for
            all_keywords: List of all keywords to search in
            top_n: Number of similar keywords to return
            
        Returns:
            List of tuples (keyword, similarity_score)
        """
        try:
            if not all_keywords:
                return []
            
            # Get all keyword texts
            keyword_texts = [k.keyword for k in all_keywords]
            
            # Add target keyword if not already present
            if target_keyword not in keyword_texts:
                keyword_texts.append(target_keyword)
            
            # Preprocess keywords
            processed_keywords = self._preprocess_keywords(keyword_texts)
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(processed_keywords)
            
            # Find target keyword index
            target_index = keyword_texts.index(target_keyword)
            
            # Calculate similarities
            similarities = cosine_similarity(
                tfidf_matrix[target_index:target_index+1], 
                tfidf_matrix
            )[0]
            
            # Create similarity pairs
            similarity_pairs = [
                (keyword_texts[i], similarities[i]) 
                for i in range(len(keyword_texts))
                if i != target_index  # Exclude the target keyword itself
            ]
            
            # Sort by similarity and return top_n
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return similarity_pairs[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar keywords: {str(e)}")
            return []
    
    def get_cluster_topics(self, clusters: List[List[str]]) -> List[str]:
        """
        Extract topic names for each cluster
        
        Args:
            clusters: List of keyword clusters
            
        Returns:
            List of topic names for each cluster
        """
        topics = []
        
        for cluster in clusters:
            if not cluster:
                topics.append("Unknown Topic")
                continue
            
            # Find most common words in the cluster
            all_words = []
            for keyword in cluster:
                words = keyword.lower().split()
                all_words.extend(words)
            
            # Count word frequencies
            word_counts = Counter(all_words)
            
            # Get most common words (excluding common stop words)
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            common_words = [
                word for word, count in word_counts.most_common(3)
                if word not in stop_words and len(word) > 2
            ]
            
            if common_words:
                topic = ' '.join(common_words).title()
            else:
                topic = cluster[0].title()  # Use first keyword as fallback
            
            topics.append(topic)
        
        return topics
