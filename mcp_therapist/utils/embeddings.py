"""
Embeddings utility module for generating and working with text embeddings.

This module provides functions for creating and manipulating text embeddings
using sentence-transformers, caching embeddings for performance, and measuring
semantic similarity between texts.
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mcp_therapist.config.settings import settings
from mcp_therapist.utils.logging import logger


class EmbeddingsManager:
    """Manager for text embeddings operations."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading the model multiple times."""
        if cls._instance is None:
            cls._instance = super(EmbeddingsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the embeddings manager."""
        if self._initialized:
            return
            
        # Configuration
        self.model_name = getattr(settings, "EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
        self.use_gpu = getattr(settings, "USE_GPU_FOR_EMBEDDINGS", True)
        self.cache_size = getattr(settings, "EMBEDDINGS_CACHE_SIZE", 1000)
        self.batch_size = getattr(settings, "EMBEDDINGS_BATCH_SIZE", 32)
        
        # Initialize the model
        try:
            device = "cuda" if self.use_gpu and torch_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded embeddings model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {str(e)}")
            logger.warning("Falling back to simple text similarity measures")
            self.model = None
        
        # Initialize embeddings cache
        self.cache = {}
        
        self._initialized = True
    
    @lru_cache(maxsize=1024)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array with the embedding vector
        """
        if self.model is None:
            # Return a random vector as fallback
            return np.random.rand(384)  # Common embedding size
        
        if not text or not isinstance(text, str):
            # Return zeros for empty/invalid input
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Generate embedding
        try:
            return self.model.encode(text, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of numpy arrays with embedding vectors
        """
        if self.model is None:
            # Return random vectors as fallback
            return [np.random.rand(384) for _ in texts]
        
        if not texts:
            return []
        
        # Filter out empty/invalid texts
        valid_texts = [t for t in texts if isinstance(t, str) and t]
        valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t]
        
        if not valid_texts:
            dim = self.model.get_sentence_embedding_dimension()
            return [np.zeros(dim) for _ in texts]
        
        # Generate embeddings
        try:
            embeddings = [None] * len(texts)
            valid_embeddings = self.model.encode(
                valid_texts, 
                batch_size=self.batch_size,
                show_progress_bar=len(valid_texts) > 100
            )
            
            # Map valid embeddings back to original indices
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = valid_embeddings[i]
            
            # Fill in zeros for invalid texts
            dim = valid_embeddings[0].shape[0]
            for i in range(len(embeddings)):
                if embeddings[i] is None:
                    embeddings[i] = np.zeros(dim)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zeros as fallback
            dim = self.model.get_sentence_embedding_dimension()
            return [np.zeros(dim) for _ in texts]
    
    def semantic_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> float:
        """Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method, one of "cosine", "dot", "euclidean"
            
        Returns:
            Similarity score (higher means more similar)
        """
        if not text1 or not text2:
            return 0.0
        
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        return vector_similarity(embedding1, embedding2, method)
    
    def semantic_clustering(
        self, texts: List[str], threshold: float = 0.85, method: str = "cosine"
    ) -> List[List[int]]:
        """Group texts into clusters based on semantic similarity.
        
        Args:
            texts: List of texts to cluster
            threshold: Similarity threshold for clustering
            method: Similarity method
            
        Returns:
            List of clusters, where each cluster is a list of indices
        """
        if not texts:
            return []
            
        # Get embeddings for all texts
        embeddings = self.get_embeddings(texts)
        
        # Cluster embeddings
        clusters = []
        used_indices = set()
        
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
                
            cluster = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(embeddings)):
                if j in used_indices:
                    continue
                
                sim = vector_similarity(embeddings[i], embeddings[j], method)
                if sim >= threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def find_semantic_outliers(
        self, texts: List[str], threshold: float = 0.6, method: str = "cosine"
    ) -> List[int]:
        """Find texts that are semantic outliers compared to the group.
        
        Args:
            texts: List of texts to analyze
            threshold: Similarity threshold below which a text is an outlier
            method: Similarity method
            
        Returns:
            List of indices of outlier texts
        """
        if len(texts) < 2:
            return []
            
        # Get embeddings for all texts
        embeddings = self.get_embeddings(texts)
        
        # Calculate average pairwise similarity for each text
        outliers = []
        for i in range(len(embeddings)):
            similarities = []
            for j in range(len(embeddings)):
                if i != j:
                    sim = vector_similarity(embeddings[i], embeddings[j], method)
                    similarities.append(sim)
            
            avg_sim = sum(similarities) / len(similarities)
            if avg_sim < threshold:
                outliers.append(i)
        
        return outliers


def torch_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def vector_similarity(v1: np.ndarray, v2: np.ndarray, method: str = "cosine") -> float:
    """Calculate similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        method: Similarity method, one of "cosine", "dot", "euclidean"
        
    Returns:
        Similarity score (higher means more similar)
    """
    if method == "cosine":
        # Cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    elif method == "dot":
        # Dot product
        return float(np.dot(v1, v2))
    
    elif method == "euclidean":
        # Euclidean distance (converted to similarity)
        dist = np.linalg.norm(v1 - v2)
        # Convert distance to similarity (1 for identical, 0 for very different)
        return float(1 / (1 + dist))
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def get_similarity(text1: str, text2: str) -> float:
    """Calculate the cosine similarity between two texts.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0 and 1
    """
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    return cosine_similarity(emb1, emb2)


# Singleton instance for global use
embeddings_manager = EmbeddingsManager() 