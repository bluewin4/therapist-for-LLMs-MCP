"""
Sentiment analysis utility module.

This module provides functions for analyzing sentiment in text, detecting
emotional tone, and identifying negativity patterns in conversation.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

from transformers import pipeline

from mcp_therapist.config.settings import settings
from mcp_therapist.utils.logging import logger


class SentimentAnalyzer:
    """Class for analyzing sentiment in text."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading the model multiple times."""
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        if self._initialized:
            return
            
        # Configuration
        self.use_sentiment = getattr(settings, "USE_SENTIMENT_ANALYSIS", True)
        self.neg_threshold = getattr(settings, "SENTIMENT_THRESHOLD_NEGATIVE", 0.3)
        self.pos_threshold = getattr(settings, "SENTIMENT_THRESHOLD_POSITIVE", 0.6)
        
        # Negative emotion keywords
        self.negative_keywords = [
            "angry", "annoyed", "awful", "bad", "confused", "disappointing", "frustrated",
            "helpless", "irritated", "misunderstood", "negative", "sad", "unhappy", "unsatisfied",
            "upset", "useless", "worthless", "terrible", "horrible", "confusing", "pointless"
        ]
        
        # Positive emotion keywords
        self.positive_keywords = [
            "amazing", "awesome", "excellent", "fantastic", "good", "great", "happy", 
            "helpful", "nice", "pleased", "positive", "satisfied", "terrific", "thank", 
            "thanks", "wonderful", "appreciated", "grateful", "impressive", "perfect"
        ]
        
        # Initialize the sentiment analysis model
        try:
            if self.use_sentiment:
                self.model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True
                )
                logger.info("Loaded sentiment analysis model")
            else:
                self.model = None
                logger.info("Sentiment analysis disabled in settings")
        except Exception as e:
            logger.error(f"Failed to load sentiment analysis model: {str(e)}")
            logger.warning("Falling back to keyword-based sentiment analysis")
            self.model = None
        
        self._initialized = True
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "negative": False,
                "positive": False
            }
        
        # Try model-based sentiment analysis first
        if self.model is not None:
            try:
                result = self.model(text[:512])  # Limit input size
                label = result[0]["label"].lower()
                score = result[0]["score"]
                
                # Transform to a consistent format
                if label == "positive":
                    sentiment_score = score
                elif label == "negative":
                    sentiment_score = 1.0 - score
                else:
                    sentiment_score = 0.5
                
                is_negative = sentiment_score < self.neg_threshold
                is_positive = sentiment_score > self.pos_threshold
                
                return {
                    "sentiment": label,
                    "score": sentiment_score,
                    "negative": is_negative,
                    "positive": is_positive,
                    "method": "model"
                }
            except Exception as e:
                logger.error(f"Error in model-based sentiment analysis: {str(e)}")
                # Fall back to keyword-based approach
        
        # Keyword-based sentiment analysis
        text_lower = text.lower()
        
        # Count negative and positive keywords
        neg_count = sum(1 for word in self.negative_keywords if word in text_lower)
        pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
        
        # Calculate sentiment score
        total_count = neg_count + pos_count
        if total_count == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = pos_count / total_count
        
        # Determine sentiment label
        if sentiment_score < self.neg_threshold:
            label = "negative"
        elif sentiment_score > self.pos_threshold:
            label = "positive"
        else:
            label = "neutral"
        
        return {
            "sentiment": label,
            "score": sentiment_score,
            "negative": sentiment_score < self.neg_threshold,
            "positive": sentiment_score > self.pos_threshold,
            "method": "keyword",
            "negative_keywords": neg_count,
            "positive_keywords": pos_count
        }
    
    def analyze_messages(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze sentiment in a list of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with aggregate sentiment analysis
        """
        if not messages:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.5,
                "negative_count": 0,
                "positive_count": 0,
                "neutral_count": 0
            }
        
        # Analyze each message
        analyses = [self.analyze_sentiment(msg) for msg in messages]
        
        # Count sentiment categories
        negative_count = sum(1 for a in analyses if a["sentiment"] == "negative")
        positive_count = sum(1 for a in analyses if a["sentiment"] == "positive")
        neutral_count = sum(1 for a in analyses if a["sentiment"] == "neutral")
        
        # Calculate average score
        average_score = sum(a["score"] for a in analyses) / len(analyses)
        
        # Determine overall sentiment
        if negative_count > positive_count:
            overall_sentiment = "negative"
        elif positive_count > negative_count:
            overall_sentiment = "positive"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": average_score,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_ratio": negative_count / len(analyses),
            "positive_ratio": positive_count / len(analyses),
            "analyses": analyses
        }
    
    def detect_sentiment_shift(self, messages: List[str], window_size: int = 3) -> Dict[str, Any]:
        """Detect shifts in sentiment over a sequence of messages.
        
        Args:
            messages: List of messages to analyze
            window_size: Size of the sliding window for shift detection
            
        Returns:
            Dictionary with sentiment shift analysis
        """
        if len(messages) < window_size * 2:
            return {
                "sentiment_shift": False,
                "shift_magnitude": 0.0
            }
        
        # Analyze each message
        analyses = [self.analyze_sentiment(msg) for msg in messages]
        scores = [a["score"] for a in analyses]
        
        # Calculate moving averages
        early_window = scores[:window_size]
        late_window = scores[-window_size:]
        
        early_avg = sum(early_window) / len(early_window)
        late_avg = sum(late_window) / len(late_window)
        
        # Calculate shift and its magnitude
        shift = late_avg - early_avg
        shift_magnitude = abs(shift)
        
        # Determine if there's a significant shift
        significant_shift = shift_magnitude > 0.2
        
        return {
            "sentiment_shift": significant_shift,
            "shift_magnitude": shift_magnitude,
            "shift_direction": "positive" if shift > 0 else "negative",
            "early_avg": early_avg,
            "late_avg": late_avg
        }


# Singleton instance for global use
sentiment_analyzer = SentimentAnalyzer() 