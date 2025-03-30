"""
Embedding generation and management for MCP Therapist.

This module provides utilities for generating and managing embeddings
for text and conversations, with caching and batching for performance.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json

from mcp_therapist.config import settings
from mcp_therapist.utils import logging
from mcp_therapist.utils.caching import memoize, get_embedding_cache
from mcp_therapist.utils.concurrency import task_manager, run_in_thread
from mcp_therapist.utils.profiling import Profiler
from mcp_therapist.core.conversation import Conversation, Message

logger = logging.get_logger(__name__)

# Conditionally import transformers based on config
if settings.embeddings.use_sentence_transformers:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence_transformers not installed, defaulting to simpler embeddings. "
            "Install with: pip install sentence-transformers"
        )
        settings.embeddings.use_sentence_transformers = False


class EmbeddingManager:
    """
    Manager for generating and handling text embeddings.
    
    This class provides methods for converting text into vector embeddings,
    with support for caching, batching, and similarity calculations.
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'EmbeddingManager':
        """
        Get the singleton instance of the embedding manager.
        
        Returns:
            EmbeddingManager instance
        """
        if cls._instance is None:
            cls._instance = EmbeddingManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.model = None
        self.model_name = settings.embeddings.model_name
        self.embedding_dim = settings.embeddings.dimension
        self.cache = get_embedding_cache()
        self.profiler = Profiler.get_instance()
        
        # Check if we should load a transformer model
        if settings.embeddings.use_sentence_transformers:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        with self.profiler.profile_section("embedding_manager.load_model"):
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(
                    f"Loaded embedding model: {self.model_name} "
                    f"(dimension: {self.embedding_dim})"
                )
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                self.model = None
                settings.embeddings.use_sentence_transformers = False
    
    @memoize(ttl=3600)  # Cache for 1 hour
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        with self.profiler.profile_section("embedding_manager.get_embedding"):
            # Normalize text
            text = text.strip()
            if not text:
                # Return zero vector for empty text
                return np.zeros(self.embedding_dim)
            
            # Generate embedding
            if settings.embeddings.use_sentence_transformers and self.model:
                embedding = self.model.encode(text)
            else:
                # Simple fallback using hash-based embedding
                embedding = self._simple_embedding(text)
            
            # Ensure it's a numpy array and normalize
            embedding = np.array(embedding)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        This method is optimized for batch processing.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        with self.profiler.profile_section("embedding_manager.get_embeddings_batch"):
            if not texts:
                return []
            
            # Check cache first
            cache_results = []
            uncached_texts = []
            uncached_indices = []
            
            # Check which texts are in cache
            for i, text in enumerate(texts):
                text = text.strip()
                cache_key = self._get_cache_key(text)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    cache_results.append((i, cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if settings.embeddings.use_sentence_transformers and self.model:
                    # Use model's batch encoding
                    uncached_embeddings = self.model.encode(uncached_texts)
                else:
                    # Parallel processing for simple embeddings
                    uncached_embeddings = task_manager.run_in_batch(
                        self._simple_embedding,
                        uncached_texts,
                        max_batch_size=settings.embeddings.batch_size or 32
                    )
            else:
                uncached_embeddings = []
            
            # Update cache with new embeddings
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                text = texts[i].strip()
                cache_key = self._get_cache_key(text)
                # Convert to numpy array and normalize
                embedding = np.array(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                self.cache.set(cache_key, embedding)
            
            # Combine cached and new embeddings
            all_embeddings = [None] * len(texts)
            
            # Fill in cached results
            for i, embedding in cache_results:
                all_embeddings[i] = embedding
            
            # Fill in new results
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                all_embeddings[i] = embedding
            
            return all_embeddings
    
    def get_message_embedding(self, message: Message) -> np.ndarray:
        """
        Generate an embedding for a message.
        
        Args:
            message: Message to generate embedding for
            
        Returns:
            Embedding vector
        """
        return self.get_embedding(message.content)
    
    def get_conversation_embedding(self, conversation: Conversation) -> np.ndarray:
        """
        Generate an embedding for a conversation.
        
        This combines embeddings of all messages, weighted by recency.
        
        Args:
            conversation: Conversation to generate embedding for
            
        Returns:
            Embedding vector
        """
        with self.profiler.profile_section("embedding_manager.get_conversation_embedding"):
            if not conversation.messages:
                return np.zeros(self.embedding_dim)
            
            # Extract message contents
            contents = [msg.content for msg in conversation.messages]
            
            # Get embeddings for all messages
            embeddings = self.get_embeddings_batch(contents)
            
            # Weight embeddings by position (more recent = higher weight)
            weights = np.linspace(
                settings.embeddings.oldest_message_weight or 0.5,
                1.0,
                len(embeddings)
            )
            
            # Combine embeddings
            weighted_sum = np.zeros(self.embedding_dim)
            for i, embedding in enumerate(embeddings):
                weighted_sum += embedding * weights[i]
            
            # Normalize
            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                weighted_sum = weighted_sum / norm
            
            return weighted_sum
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        with self.profiler.profile_section("embedding_manager.calculate_similarity"):
            # Reshape for sklearn
            e1 = embedding1.reshape(1, -1)
            e2 = embedding2.reshape(1, -1)
            
            # Calculate similarity
            sim = cosine_similarity(e1, e2)[0][0]
            
            # Ensure it's in valid range
            return max(0.0, min(1.0, float(sim)))
    
    def calculate_similarities(
        self,
        reference: np.ndarray,
        candidates: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate similarities between a reference and multiple candidates.
        
        This is optimized for batch comparison.
        
        Args:
            reference: Reference embedding
            candidates: List of candidate embeddings
            
        Returns:
            List of similarity scores
        """
        with self.profiler.profile_section("embedding_manager.calculate_similarities"):
            if not candidates:
                return []
            
            # Reshape reference for sklearn
            ref = reference.reshape(1, -1)
            
            # Stack candidates into a matrix
            cand_matrix = np.vstack([c.reshape(1, -1) for c in candidates])
            
            # Calculate similarities
            sims = cosine_similarity(ref, cand_matrix)[0]
            
            # Ensure values are in valid range
            return [max(0.0, min(1.0, float(s))) for s in sims]
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simple embedding for the given text.
        
        This is a fallback when transformer models aren't available.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode('utf-8')).digest()
        
        # Convert hash to a list of floats
        hash_values = [float(b) / 255.0 for b in text_hash]
        
        # Pad or truncate to the desired dimension
        if len(hash_values) < self.embedding_dim:
            hash_values.extend([0.0] * (self.embedding_dim - len(hash_values)))
        elif len(hash_values) > self.embedding_dim:
            hash_values = hash_values[:self.embedding_dim]
        
        return np.array(hash_values)
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key string
        """
        # Include model name in key to avoid conflicts if model changes
        key_data = {
            "text": text,
            "model": self.model_name,
            "dim": self.embedding_dim
        }
        
        # Create a stable JSON representation
        key_json = json.dumps(key_data, sort_keys=True)
        
        # Hash the JSON string
        return hashlib.md5(key_json.encode('utf-8')).hexdigest()


# Create a global instance
embedding_manager = EmbeddingManager.get_instance() 