"""
Utility module for topic analysis and detection for the MCP Therapist system.
Provides functionality for TF-IDF processing, topic extraction, and topic fixation detection.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TopicAnalyzer:
    """
    Provides utilities for analyzing topics in conversations using TF-IDF and other methods.
    """
    
    def __init__(self, min_term_freq: int = 2, top_n_terms: int = 10, max_similarity_threshold: float = 0.7):
        """
        Initialize the TopicAnalyzer with configurable parameters.
        
        Args:
            min_term_freq: Minimum frequency required for a term to be considered significant
            top_n_terms: Number of top terms to extract from each message
            max_similarity_threshold: Threshold for determining high topic similarity
        """
        self.min_term_freq = min_term_freq
        self.top_n_terms = top_n_terms
        self.max_similarity_threshold = max_similarity_threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=0.01,
            max_features=100
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic analysis by removing special characters,
        lemmatizing, and removing stopwords.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def extract_topics(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extract the most significant topics from each text using TF-IDF.
        
        Args:
            texts: List of text messages
            
        Returns:
            List of dictionaries mapping top terms to their TF-IDF scores for each text
        """
        if not texts:
            return []
            
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit and transform the TF-IDF vectorizer
        try:
            tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        except ValueError:
            # Handle case when texts are too short or contain only stopwords
            return [{} for _ in texts]
            
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract top terms for each text
        topics = []
        for i in range(len(texts)):
            # Get TF-IDF scores for current text
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            
            # Create a dictionary of term -> score
            term_scores = {feature_names[j]: tfidf_scores[j] for j in range(len(feature_names))}
            
            # Sort by score and take top N
            top_terms = dict(sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n_terms])
            
            topics.append(top_terms)
            
        return topics
    
    def calculate_topic_overlap(self, topics1: Dict[str, float], topics2: Dict[str, float]) -> float:
        """
        Calculate the overlap between two topic sets.
        
        Args:
            topics1: First topic dictionary (term -> score)
            topics2: Second topic dictionary (term -> score)
            
        Returns:
            Similarity score (0-1) representing topic overlap
        """
        # Get sets of terms
        terms1 = set(topics1.keys())
        terms2 = set(topics2.keys())
        
        # Calculate Jaccard similarity
        if not terms1 or not terms2:
            return 0.0
            
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_topic_similarity(self, texts: List[str]) -> List[float]:
        """
        Calculate rolling similarity between consecutive texts in a conversation.
        
        Args:
            texts: List of text messages in chronological order
            
        Returns:
            List of similarity scores between consecutive texts
        """
        if len(texts) < 2:
            return []
            
        topics = self.extract_topics(texts)
        similarities = []
        
        for i in range(1, len(topics)):
            sim = self.calculate_topic_overlap(topics[i-1], topics[i])
            similarities.append(sim)
            
        return similarities
    
    def detect_topic_fixation(self, texts: List[str], window_size: int = 4, 
                             similarity_threshold: float = 0.6) -> Tuple[bool, float, Optional[List[str]]]:
        """
        Detect if a conversation is fixated on specific topics.
        
        Args:
            texts: List of text messages in chronological order
            window_size: Number of messages to consider for fixation detection
            similarity_threshold: Threshold above which similarity indicates fixation
            
        Returns:
            Tuple containing:
            - Boolean indicating whether topic fixation is detected
            - Confidence score (0-1)
            - List of the most repeated terms (or None if no fixation)
        """
        if len(texts) < window_size:
            return False, 0.0, None
            
        # Focus on the most recent messages
        recent_texts = texts[-window_size:]
        
        # Extract topics from recent messages
        topics = self.extract_topics(recent_texts)
        
        # Calculate average similarity between consecutive messages
        similarities = self.calculate_topic_similarity(recent_texts)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Identify common terms across messages
        term_counter = Counter()
        for topic_dict in topics:
            term_counter.update(topic_dict.keys())
            
        # Get terms that appear in at least min_term_freq messages
        repeated_terms = [term for term, count in term_counter.items() 
                         if count >= self.min_term_freq]
        
        # Calculate confidence based on similarity and term repetition
        confidence = avg_similarity * (len(repeated_terms) / (self.top_n_terms * 2))
        
        # Determine if there's topic fixation
        is_fixated = avg_similarity >= similarity_threshold and repeated_terms
        
        return is_fixated, confidence, repeated_terms if is_fixated else None 