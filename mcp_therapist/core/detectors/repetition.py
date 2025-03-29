"""
Detector for identifying repetition patterns in a conversation.

This module provides a detector that monitors conversations for signs of repetition,
where the assistant repeats the same or similar responses across multiple turns.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Any

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult
from mcp_therapist.utils.logging import logger
from mcp_therapist.utils.text import (
    find_common_phrases,
    preprocess_text,
    similarity_score,
    calculate_similarity
)


class RepetitionDetector(BaseDetector):
    """Detector for identifying repetition patterns in conversations.
    
    This detector analyzes conversations to identify when the assistant is repeating
    the same or similar content across messages. Repetition can manifest as:
    1. Exact duplication of phrases or sentences
    2. Semantic repetition (similar meaning expressed in different words)
    3. Structural repetition (same patterns or templates reused)
    """
    
    def __init__(self, threshold: float = 0.5, window_size: int = 3, min_messages: int = 2):
        """Initialize the repetition detector.
        
        Args:
            threshold: The similarity threshold for detecting repetition.
            window_size: The number of recent messages to analyze.
            min_messages: The minimum number of messages required for analysis.
        """
        super().__init__()
        self.threshold = getattr(settings, "REPETITION_THRESHOLD", threshold)
        self.window_size = getattr(settings, "REPETITION_WINDOW_SIZE", window_size)
        self.min_messages = min_messages
        
        logger.info(f"Repetition detector initialized: threshold={self.threshold}, window_size={self.window_size}")
    
    @property
    def detector_type(self) -> str:
        """Return the type of detector."""
        return "repetition"
    
    @property
    def rut_type(self) -> RutType:
        """Return the type of rut this detector identifies."""
        return RutType.REPETITION
    
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """Analyze the conversation for signs of repetition.
        
        Args:
            conversation: The conversation to analyze.
                
        Returns:
            RutAnalysisResult with repetition detection data.
        """
        messages = conversation.get_recent_messages(self.window_size)
        
        # Filter to only assistant messages
        assistant_messages = [msg for msg in messages if msg.role == MessageRole.ASSISTANT]
        
        # If we don't have enough messages, return early
        if len(assistant_messages) < self.min_messages:
            logger.debug(f"Not enough assistant messages to analyze for repetition: {len(assistant_messages)}")
            return RutAnalysisResult(
                conversation_id=conversation.id,
                rut_detected=False,
                rut_type=RutType.REPETITION,
                confidence=0.0,
                evidence={
                    "message_count": len(assistant_messages),
                    "min_required": self.min_messages
                }
            )
        
        # Check different types of repetition
        phrase_repetition = self._detect_exact_phrase_repetition(assistant_messages)
        semantic_repetition = self._detect_semantic_repetition(assistant_messages)
        structural_repetition = self._detect_structural_repetition(assistant_messages)
        
        # Combine evidence and calculate overall confidence
        evidence = {}
        confidence = 0.0
        
        if phrase_repetition["score"] > 0:
            confidence = max(confidence, phrase_repetition["score"])
            evidence["phrase_repetition"] = phrase_repetition["phrases"]
        
        if semantic_repetition["score"] > 0:
            confidence = max(confidence, semantic_repetition["score"])
            evidence["semantic_repetition"] = semantic_repetition["similarity_scores"]
        
        if structural_repetition["score"] > 0:
            confidence = max(confidence, structural_repetition["score"])
            evidence["structural_repetition"] = structural_repetition["patterns"]
        
        # Determine if repetition is detected based on confidence threshold
        rut_detected = confidence >= self.threshold
        
        logger.debug(f"Repetition analysis complete: detected={rut_detected}, confidence={confidence}")
        
        return RutAnalysisResult(
            conversation_id=conversation.id,
            rut_detected=rut_detected,
            rut_type=RutType.REPETITION,
            confidence=confidence,
            evidence=evidence
        )
    
    def _detect_exact_phrase_repetition(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect exact phrases that are repeated across messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with repetition score and evidence
        """
        result = {"score": 0.0, "phrases": []}
        
        # Extract content from messages
        message_texts = [msg.content for msg in messages]
        
        # Generate n-grams for each message
        all_ngrams = []
        for text in message_texts:
            # Preprocess the text
            processed_text = preprocess_text(text)
            words = processed_text.split()
            
            # Generate n-grams
            for n in range(self.min_ngram_length, min(self.max_ngram_length + 1, len(words) + 1)):
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)
        
        # Count occurrences of each n-gram
        ngram_counts = Counter(all_ngrams)
        
        # Find repeated phrases
        repeated_phrases = []
        for phrase, count in ngram_counts.items():
            if count > 1 and len(phrase.split()) >= self.min_repeated_phrase_length:
                repeated_phrases.append((phrase, count))
        
        # Sort by count (most repeated first)
        repeated_phrases.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate a score based on repeated phrases
        if repeated_phrases:
            # Extract just the phrases
            result["phrases"] = [phrase for phrase, count in repeated_phrases]
            
            # Calculate score based on repetition count and phrase length
            repetition_severity = sum(count * len(phrase.split()) for phrase, count in repeated_phrases)
            total_words = sum(len(preprocess_text(msg.content).split()) for msg in messages)
            
            if total_words > 0:
                result["score"] = min(1.0, repetition_severity / (2 * total_words))
        
        return result
    
    def _detect_semantic_repetition(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect semantic similarity across messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with repetition score and evidence
        """
        result = {"score": 0.0, "similarity_scores": []}
        
        if len(messages) < 2:
            return result
        
        # Calculate pairwise similarity between all messages
        similarity_scores = []
        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                text1 = messages[i].content
                text2 = messages[j].content
                
                similarity = calculate_similarity(text1, text2)
                if similarity >= self.similarity_threshold:
                    similarity_scores.append({
                        "message1_idx": i,
                        "message2_idx": j,
                        "similarity": similarity
                    })
        
        # Sort by similarity (highest first)
        similarity_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate score based on similarity scores
        if similarity_scores:
            result["similarity_scores"] = similarity_scores
            
            # Average of top similarities, weighted by how many message pairs have high similarity
            avg_similarity = sum(score["similarity"] for score in similarity_scores) / len(similarity_scores)
            coverage = min(1.0, len(similarity_scores) / (len(messages) * (len(messages) - 1) / 2))
            
            result["score"] = avg_similarity * coverage
        
        return result
    
    def _detect_structural_repetition(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect repeated structural patterns in messages.
        
        This looks for repeated sentence structures, openings, and closings.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with repetition score and evidence
        """
        result = {"score": 0.0, "patterns": {}}
        
        # Extract starts and ends of messages
        starts = []
        ends = []
        
        for msg in messages:
            text = msg.content
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                starts.append(sentences[0])
                ends.append(sentences[-1])
        
        # Check for repetitive starts
        start_counter = Counter(starts)
        repeated_starts = [(start, count) for start, count in start_counter.items() if count > 1]
        
        # Check for repetitive ends
        end_counter = Counter(ends)
        repeated_ends = [(end, count) for end, count in end_counter.items() if count > 1]
        
        # Collect evidence
        if repeated_starts:
            result["patterns"]["repeated_starts"] = [start for start, _ in repeated_starts]
        
        if repeated_ends:
            result["patterns"]["repeated_ends"] = [end for end, _ in repeated_ends]
        
        # Calculate score
        if repeated_starts or repeated_ends:
            start_score = sum(count / len(messages) for _, count in repeated_starts)
            end_score = sum(count / len(messages) for _, count in repeated_ends)
            
            result["score"] = (start_score + end_score) / 2
        
        return result 