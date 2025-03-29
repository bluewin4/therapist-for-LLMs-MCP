"""
Detector for identifying stagnation in a conversation.

This module provides a detector that monitors conversations for signs of stagnation,
where the conversation is not making substantive progress or meaningful development.
"""

import re
from typing import Dict, List, Optional, Set, Any

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.models.conversation import Conversation, Message, RutType, RutAnalysisResult
from mcp_therapist.utils.logging import logger
from mcp_therapist.utils.text import preprocess_text, similarity_score, calculate_similarity


class StagnationDetector(BaseDetector):
    """Detector for identifying stagnation in conversations.
    
    Stagnation occurs when a conversation lacks meaningful development or progress despite
    continuing exchanges. This can manifest as:
    1. Low semantic diversity across consecutive messages (similar content recycled)
    2. Circular references where topics repeat without advancement
    3. Extended exchanges with minimal new informational content
    """
    
    def __init__(self, threshold: float = 0.6, window_size: int = 3, min_messages: int = 3):
        """Initialize the stagnation detector.
        
        Args:
            threshold: The similarity threshold for detecting stagnation.
            window_size: The number of recent messages to analyze.
            min_messages: The minimum number of messages required for analysis.
        """
        super().__init__()
        self.threshold = getattr(settings, "STAGNATION_THRESHOLD", threshold)
        self.window_size = getattr(settings, "STAGNATION_WINDOW_SIZE", window_size)
        self.min_messages = min_messages
        self.embedding_cache = {}  # Cache for embeddings to avoid recomputation
        
        # Configuration
        self.confidence_threshold = getattr(settings, "STAGNATION_CONFIDENCE_THRESHOLD", 0.6)
        self.time_threshold = getattr(settings, "STAGNATION_TIME_THRESHOLD", 300)  # 5 minutes
        self.topic_similarity_threshold = getattr(settings, "TOPIC_SIMILARITY_THRESHOLD", 0.8)
        self.progress_indicator_threshold = getattr(settings, "PROGRESS_INDICATOR_THRESHOLD", 0.3)
        
        # Stagnation indicators
        self.filler_phrase_patterns = [
            r"as (I've|I have) (mentioned|said|stated|noted|pointed out)( before)?",
            r"(like|as) I said( before)?",
            r"(I'm|I am) (not sure|uncertain|confused) (what|how) (else|more) (to|I can)",
            r"we('ve| have) (already|previously) (discussed|covered|gone over|talked about)",
            r"(to|let me) (reiterate|summarize|repeat)",
            r"(I|we) (seem to be|are) going (around|in) circles",
            r"(I'm|I am) not sure (how|where) to (proceed|go) (from here|next)",
            r"(I|we) (don't|do not) (seem to be|appear to be) making (progress|headway)",
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.filler_phrase_patterns
        ]
        
        # Topic keywords - Maintain state across analysis calls
        self.recent_topics: Set[str] = set()
        self.topic_repetition_count: Dict[str, int] = {}
        
        logger.info(f"Stagnation detector initialized: threshold={self.threshold}, window_size={self.window_size}")
    
    @property
    def detector_type(self) -> str:
        """Return the type of detector."""
        return "stagnation"
    
    @property
    def rut_type(self) -> RutType:
        """Return the type of rut this detector identifies."""
        return RutType.STAGNATION
    
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """Analyze the conversation for signs of stagnation.
        
        Args:
            conversation: The conversation to analyze.
                
        Returns:
            RutAnalysisResult with stagnation detection data.
        """
        messages = conversation.get_recent_messages(self.window_size)
        
        # If we don't have enough messages, return early
        if len(messages) < self.min_messages:
            logger.debug(f"Not enough messages to analyze for stagnation: {len(messages)}")
            return RutAnalysisResult(
                conversation_id=conversation.id,
                rut_detected=False,
                rut_type=RutType.STAGNATION,
                confidence=0.0,
                evidence={
                    "message_count": len(messages),
                    "min_required": self.min_messages
                }
            )
        
        # Check different types of stagnation
        time_stagnation = self._detect_time_stagnation(messages)
        topic_stagnation = self._detect_topic_stagnation(messages)
        progress_indicators = self._detect_progress_indicators(messages)
        
        # Combine evidence and calculate overall confidence
        evidence = {}
        confidence = 0.0
        
        if time_stagnation["score"] > 0:
            confidence = max(confidence, time_stagnation["score"])
            evidence["time_gaps"] = time_stagnation["gaps"]
        
        if topic_stagnation["score"] > 0:
            confidence = max(confidence, topic_stagnation["score"])
            evidence["topic_similarity"] = topic_stagnation["similarities"]
        
        if progress_indicators["score"] > 0:
            confidence = max(confidence, progress_indicators["score"])
            evidence["progress_indicators"] = progress_indicators["indicators"]
        
        # Determine if stagnation is detected based on confidence threshold
        rut_detected = confidence >= self.confidence_threshold
        
        logger.debug(f"Stagnation analysis complete: detected={rut_detected}, confidence={confidence}")
        
        return RutAnalysisResult(
            conversation_id=conversation.id,
            rut_detected=rut_detected,
            rut_type=RutType.STAGNATION,
            confidence=confidence,
            evidence=evidence
        )
    
    def _detect_time_stagnation(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect stagnation based on slowing conversation pace.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with stagnation score and evidence
        """
        result = {"score": 0.0, "gaps": []}
        
        if len(messages) < 3:
            return result
        
        # Calculate time gaps between consecutive messages
        time_gaps = []
        for i in range(1, len(messages)):
            current_time = messages[i].timestamp
            previous_time = messages[i-1].timestamp
            
            gap = current_time - previous_time
            time_gaps.append(gap)
        
        # Look for increasing time gaps
        increasing_gaps = 0
        significant_gaps = 0
        
        for i in range(1, len(time_gaps)):
            if time_gaps[i] > time_gaps[i-1] * 1.5:  # 50% increase
                increasing_gaps += 1
            
            if time_gaps[i] > self.time_threshold:
                significant_gaps += 1
        
        # Calculate score based on gap patterns
        if time_gaps:
            # Record evidence
            result["gaps"] = [{"index": i, "gap": gap} for i, gap in enumerate(time_gaps)]
            
            # Calculate score components
            increasing_score = increasing_gaps / max(1, len(time_gaps) - 1)
            significant_score = significant_gaps / len(time_gaps)
            
            # Combined score
            result["score"] = max(increasing_score, significant_score)
        
        return result
    
    def _detect_topic_stagnation(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect stagnation based on circling the same topic.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with stagnation score and evidence
        """
        result = {"score": 0.0, "similarities": []}
        
        if len(messages) < 3:
            return result
        
        # Get all user messages
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        
        # Skip if not enough user messages
        if len(user_messages) < 2:
            return result
        
        # Calculate semantic similarity between non-adjacent user messages
        similarities = []
        for i in range(len(user_messages)):
            for j in range(i + 2, len(user_messages)):  # Skip adjacent messages
                text1 = user_messages[i].content
                text2 = user_messages[j].content
                
                similarity = calculate_similarity(text1, text2)
                if similarity >= self.topic_similarity_threshold:
                    similarities.append({
                        "message1_idx": i,
                        "message2_idx": j,
                        "similarity": similarity
                    })
        
        # Calculate stagnation score based on similarities
        if similarities:
            result["similarities"] = similarities
            
            # Average similarity of topic circles
            avg_similarity = sum(s["similarity"] for s in similarities) / len(similarities)
            
            # Weight by how many message pairs show high similarity
            coverage = min(1.0, len(similarities) / (len(user_messages) * (len(user_messages) - 1) / 4))
            
            result["score"] = avg_similarity * coverage
        
        return result
    
    def _detect_progress_indicators(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect stagnation based on lack of progress indicators.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with stagnation score and evidence
        """
        result = {"score": 0.0, "indicators": {}}
        
        # Define progress and stagnation indicators
        progress_phrases = [
            "i understand", "that makes sense", "now i get it", "i see", 
            "thank you", "helpful", "perfect", "great", "excellent"
        ]
        
        stagnation_phrases = [
            "still confused", "don't understand", "not clear", "unclear",
            "you said earlier", "you already said", "we discussed this",
            "going in circles", "same thing", "repeating", "not helping"
        ]
        
        # Count indicators in user messages
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        
        if not user_messages:
            return result
        
        # Count phrase occurrences
        progress_count = 0
        stagnation_count = 0
        
        for msg in user_messages:
            content = msg.content.lower()
            
            for phrase in progress_phrases:
                if phrase in content:
                    progress_count += 1
            
            for phrase in stagnation_phrases:
                if phrase in content:
                    stagnation_count += 1
        
        # Calculate score
        total_indicators = progress_count + stagnation_count
        
        if total_indicators > 0:
            # Record evidence
            result["indicators"] = {
                "progress_count": progress_count,
                "stagnation_count": stagnation_count,
                "progress_ratio": progress_count / max(1, total_indicators)
            }
            
            # Stagnation score is high when progress indicators are low
            progress_ratio = progress_count / max(1, total_indicators)
            
            if progress_ratio < self.progress_indicator_threshold:
                # Low progress ratio indicates stagnation
                result["score"] = min(1.0, (self.progress_indicator_threshold - progress_ratio) / self.progress_indicator_threshold)
        
        return result
    
    def _analyze_semantic_diversity(self, processed_texts: List[str]) -> Dict:
        """Analyze the semantic diversity of recent messages.
        
        Low semantic diversity indicates possible stagnation.
        
        Args:
            processed_texts: List of preprocessed message texts.
            
        Returns:
            Dictionary with diversity score and supporting data.
        """
        if len(processed_texts) < 2:
            return {"score": 0.0, "explanation": "Not enough messages to analyze diversity"}
        
        # Calculate pairwise similarities between consecutive messages
        similarities = []
        for i in range(1, len(processed_texts)):
            sim = similarity_score(processed_texts[i-1], processed_texts[i])
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # High similarity means low diversity, so invert the score
        diversity_score = min(1.0, max(0.0, avg_similarity))
        
        return {
            "score": diversity_score,
            "average_similarity": avg_similarity,
            "individual_similarities": similarities,
            "explanation": "High message similarity indicates low semantic diversity"
        }
    
    def _detect_filler_phrases(self, messages: List[Message]) -> Dict:
        """Detect filler phrases that indicate stagnation.
        
        Args:
            messages: List of messages to analyze.
            
        Returns:
            Dictionary with filler phrase detection score and matches.
        """
        matches = []
        
        for msg in messages:
            for pattern in self.compiled_patterns:
                found = pattern.findall(msg.content)
                if found:
                    matches.extend([
                        {
                            "pattern": pattern.pattern,
                            "match": match,
                            "message_id": msg.id
                        } for match in found
                    ])
        
        # Calculate score based on number of matches relative to number of messages
        score = min(1.0, len(matches) / len(messages) * 1.5) if messages else 0.0
        
        return {
            "score": score,
            "matches": matches,
            "count": len(matches),
            "explanation": "Presence of filler phrases indicates conversation stagnation"
        }
    
    def _detect_topic_repetition(self, processed_texts: List[str]) -> Dict:
        """Detect repetition of topics without advancement.
        
        Args:
            processed_texts: List of preprocessed message texts.
            
        Returns:
            Dictionary with topic repetition score and supporting data.
        """
        # Extract potential topics using keyword frequency
        current_topics = set()
        for text in processed_texts:
            # Simple approach: split by spaces and take words longer than 4 chars
            words = text.split()
            for word in words:
                if len(word) > 4:
                    current_topics.add(word.lower())
        
        # Identify repeated topics from previous analyses
        repeated_topics = self.recent_topics.intersection(current_topics)
        
        # Update topic repetition counts
        for topic in repeated_topics:
            self.topic_repetition_count[topic] = self.topic_repetition_count.get(topic, 0) + 1
        
        # Clean up old topics
        for topic in list(self.topic_repetition_count.keys()):
            if topic not in current_topics:
                self.topic_repetition_count.pop(topic, None)
        
        # Update recent topics
        self.recent_topics = current_topics
        
        # Calculate repetition score
        if not self.recent_topics:
            repetition_score = 0.0
        else:
            # Score is ratio of repeated topics to total topics, adjusted by repetition count
            repetition_sum = sum(self.topic_repetition_count.values())
            total_topics = len(self.recent_topics)
            repetition_score = min(1.0, (repetition_sum / (total_topics * 2)) * 0.8)
        
        return {
            "score": repetition_score,
            "repeated_topics": list(repeated_topics),
            "repetition_counts": self.topic_repetition_count,
            "explanation": "Repetition of topics without advancement indicates stagnation"
        } 