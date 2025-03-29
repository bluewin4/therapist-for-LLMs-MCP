"""
Module for detecting contradictions in conversations.
Identifies when an LLM contradicts its previous statements.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import re
import numpy as np
from datetime import datetime

from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.models.conversation import RutType, Message, MessageRole, Conversation, RutAnalysisResult
from mcp_therapist.utils.embeddings import EmbeddingsManager
from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.registry import register_detector
from mcp_therapist.utils.logging import logger

logger = logging.getLogger(__name__)

@register_detector
class ContradictionDetector(BaseDetector):
    """
    Detector for identifying contradictions in assistant responses.
    Uses both keyword-based and semantic methods to detect contradictory statements.
    """
    
    def __init__(self, 
                min_messages: int = 4,
                contradiction_threshold: float = 0.8,
                window_size: int = 10):
        """
        Initialize the contradiction detector.
        
        Args:
            min_messages: Minimum number of messages required before analysis
            contradiction_threshold: Threshold for semantic contradiction detection
            window_size: Number of recent messages to analyze
        """
        super().__init__()
        self.min_messages = min_messages
        self.contradiction_threshold = getattr(settings, "CONTRADICTION_THRESHOLD", contradiction_threshold)
        self.window_size = window_size
        self.embeddings_manager = EmbeddingsManager()
        
        # Common negation patterns
        self.negation_patterns = [
            r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bcan't\b", r"\bcannot\b",
            r"\bwon't\b", r"\bdidn't\b", r"\bisn't\b", r"\baren't\b", r"\bwasn't\b",
            r"\bweren't\b", r"\bdon't\b", r"\bdoesn't\b", r"\bshouldn't\b",
            r"\bcouldn't\b", r"\bwouldn't\b", r"\bincorrect\b", r"\buntrue\b"
        ]
        
        # Contrasting transition phrases
        self.contrasting_phrases = [
            r"\bhowever\b", r"\bbut\b", r"\byet\b", r"\balthough\b", r"\beven though\b",
            r"\bon the contrary\b", r"\bin contrast\b", r"\binstead\b", r"\bconversely\b",
            r"\bnonetheless\b", r"\bnevertheless\b", r"\bdespite\b", r"\bin fact\b",
            r"\bactually\b", r"\bon second thought\b", r"\bi was wrong\b", r"\bincorrectly stated\b"
        ]
        
        # Polar opposite term pairs (simple examples)
        self.polar_terms = {
            "true": "false",
            "yes": "no",
            "always": "never",
            "all": "none",
            "everyone": "no one",
            "everything": "nothing",
            "can": "cannot",
            "possible": "impossible",
            "allow": "disallow",
            "permitted": "forbidden",
            "legal": "illegal",
            "safe": "unsafe",
            "right": "wrong",
            "correct": "incorrect",
        }
        
        # Patterns for identifying statements in text
        self.statement_patterns = [
            r"(I am [^.!?]*)[.!?]",  # I am statements
            r"(You are [^.!?]*)[.!?]",  # You are statements
            r"(It is [^.!?]*)[.!?]",  # It is statements
            r"(There (?:is|are) [^.!?]*)[.!?]",  # There is/are statements
            r"(The [^.!?]* (?:is|are) [^.!?]*)[.!?]",  # The X is/are statements
            r"([^.!?]* (?:always|never) [^.!?]*)[.!?]",  # Always/never statements
            r"([A-Z][^.!?]* (?:can|cannot|can't) [^.!?]*)[.!?]"  # Can/cannot statements
        ]
        
        # Compile statement patterns
        self.compiled_statement_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.statement_patterns]
        
        logger.info(f"Contradiction detector initialized with threshold {self.contradiction_threshold}")
    
    @property
    def detector_type(self) -> str:
        """Get the type of this detector."""
        return "contradiction"
        
    @property
    def rut_type(self) -> RutType:
        """Get the rut type that this detector identifies."""
        return RutType.CONTRADICTION
        
    def _extract_assistant_messages(self, conversation: Conversation) -> List[Message]:
        """
        Extract assistant messages from the conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            List of assistant messages
        """
        return [msg for msg in conversation.messages 
                if msg.role == MessageRole.ASSISTANT and msg.content.strip()]
                
    def _identify_statements(self, text: str) -> List[str]:
        """
        Break text into individual statements (sentences or clauses).
        
        Args:
            text: The text to analyze
            
        Returns:
            List of statements
        """
        statements = []
        
        # Apply each pattern to find statements
        for pattern in self.compiled_statement_patterns:
            matches = pattern.findall(text)
            statements.extend(matches)
            
        # Also split by periods and extract sentences that are statements
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        for sentence in sentences:
            # Only include sentences that make an assertion
            if any(assertion in sentence.lower() for assertion in [
                "is", "are", "was", "were", "will", "can", "cannot", "should", "must", 
                "always", "never", "definitely", "certainly", "absolutely"
            ]):
                if sentence not in statements:
                    statements.append(sentence)
                    
        return statements
    
    def _check_direct_contradictions(self, messages: List[Message]) -> List[Tuple[str, str, float]]:
        """
        Check for direct contradictions using keyword analysis.
        
        Args:
            messages: The messages to analyze
            
        Returns:
            List of tuples with contradictory statements and confidence
        """
        contradictions = []
        
        # Extract statements from messages
        message_statements = [
            (i, self._identify_statements(msg.content))
            for i, msg in enumerate(messages)
        ]
        
        # Check each message against earlier messages
        for i in range(1, len(message_statements)):
            current_msg_idx, current_statements = message_statements[i]
            
            # Look through statements in the current message
            for current_stmt in current_statements:
                current_lower = current_stmt.lower()
                
                # Look for negation patterns and contrasting phrases
                has_negation = any(re.search(pattern, current_lower) for pattern in self.negation_patterns)
                has_contrast = any(re.search(pattern, current_lower) for pattern in self.contrasting_phrases)
                
                # If the statement has negation or contrasting patterns, 
                # check against earlier messages for potential contradictions
                if has_negation or has_contrast:
                    for j in range(i):
                        prev_msg_idx, prev_statements = message_statements[j]
                        
                        for prev_stmt in prev_statements:
                            # Check for semantic similarity with negation = potential contradiction
                            similarity = self.embeddings_manager.calculate_semantic_similarity(
                                prev_stmt, current_stmt
                            )
                            
                            # If similar content but with negation/contrast,
                            # it's likely a contradiction
                            if similarity > 0.7 and (has_negation or has_contrast):
                                confidence = similarity * (1.2 if has_negation and has_contrast else 1.0)
                                contradictions.append((prev_stmt, current_stmt, min(confidence, 1.0)))
                                
                # Check for polar opposite terms
                for term, opposite in self.polar_terms.items():
                    if term in current_lower:
                        # Look for the opposite term in previous statements
                        for j in range(i):
                            prev_msg_idx, prev_statements = message_statements[j]
                            for prev_stmt in prev_statements:
                                if opposite in prev_stmt.lower():
                                    # Calculate confidence based on context similarity
                                    similarity = self.embeddings_manager.calculate_semantic_similarity(
                                        prev_stmt, current_stmt
                                    )
                                    if similarity > 0.6:  # If contexts are similar
                                        contradictions.append((prev_stmt, current_stmt, min(similarity * 1.2, 1.0)))
        
        return contradictions
    
    def _check_factual_inconsistencies(self, messages: List[Message]) -> List[Tuple[str, str, float]]:
        """
        Check for factual inconsistencies using semantic analysis.
        
        Args:
            messages: The messages to analyze
            
        Returns:
            List of tuples with inconsistent statements and confidence
        """
        inconsistencies = []
        
        # Extract statements from messages
        message_statements = [
            (i, self._identify_statements(msg.content))
            for i, msg in enumerate(messages)
        ]
        
        # Compare statements across messages for contradictions
        for i in range(1, len(message_statements)):
            current_msg_idx, current_statements = message_statements[i]
            
            for current_stmt in current_statements:
                # Skip very short statements that could lead to false positives
                if len(current_stmt.split()) < 5:
                    continue
                    
                for j in range(i):
                    prev_msg_idx, prev_statements = message_statements[j]
                    
                    for prev_stmt in prev_statements:
                        # Skip very short statements that could lead to false positives
                        if len(prev_stmt.split()) < 5:
                            continue
                            
                        # Calculate semantic contradiction
                        # (high cosine similarity with high negation indicators)
                        similarity = self.embeddings_manager.calculate_semantic_similarity(
                            prev_stmt, current_stmt
                        )
                        
                        # Check for inconsistency indicators
                        current_lower = current_stmt.lower()
                        prev_lower = prev_stmt.lower()
                        
                        # Count negation terms
                        current_negations = sum(1 for pattern in self.negation_patterns 
                                             if re.search(pattern, current_lower))
                        prev_negations = sum(1 for pattern in self.negation_patterns 
                                           if re.search(pattern, prev_lower))
                        
                        # Different number of negations in similar statements might indicate contradiction
                        if similarity > 0.8 and current_negations != prev_negations:
                            confidence = similarity * min(1.0, 0.7 + 0.1 * abs(current_negations - prev_negations))
                            inconsistencies.append((prev_stmt, current_stmt, confidence))
                            
                        # Very similar statements with opposite polarity
                        elif similarity > self.contradiction_threshold:
                            # Extract key phrases to check for polarity
                            prev_phrases = set(re.findall(r'\b\w+\b', prev_lower))
                            current_phrases = set(re.findall(r'\b\w+\b', current_lower))
                            
                            # Check for term/opposite term pairs
                            for term, opposite in self.polar_terms.items():
                                if (term in prev_phrases and opposite in current_phrases) or \
                                   (term in current_phrases and opposite in prev_phrases):
                                    confidence = similarity * 1.1
                                    inconsistencies.append((prev_stmt, current_stmt, min(confidence, 1.0)))
        
        return inconsistencies
    
    def detect(self, conversation: Conversation) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect contradictions in a conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Tuple containing:
            - Boolean indicating whether contradictions were detected
            - Confidence score (0-1)
            - Dictionary with additional detection details
        """
        # Check if there are enough messages
        assistant_messages = self._extract_assistant_messages(conversation)
        if len(assistant_messages) < self.min_messages:
            logger.debug(f"Not enough assistant messages ({len(assistant_messages)}) for contradiction detection")
            return False, 0.0, {"reason": "Not enough messages"}
        
        # Limit to the most recent window
        recent_messages = assistant_messages[-self.window_size:]
        
        # Check for direct contradictions
        direct_contradictions = self._check_direct_contradictions(recent_messages)
        
        # Check for factual inconsistencies
        factual_inconsistencies = self._check_factual_inconsistencies(recent_messages)
        
        # Combine all detected contradictions
        all_contradictions = direct_contradictions + factual_inconsistencies
        
        # Sort by confidence (highest first)
        all_contradictions.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate overall confidence
        if all_contradictions:
            # Use the highest confidence contradiction as the main confidence
            confidence = all_contradictions[0][2]
            
            # Prepare evidence
            contradiction_evidence = [
                {
                    "statement1": stmt1,
                    "statement2": stmt2,
                    "confidence": conf
                }
                for stmt1, stmt2, conf in all_contradictions[:3]  # Top 3 contradictions
            ]
            
            logger.info(f"Contradiction detected with confidence {confidence:.2f}")
            return True, confidence, {
                "contradictions": contradiction_evidence,
                "message_count": len(recent_messages)
            }
        
        return False, 0.0, {"reason": "No contradictions detected"}
    
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """
        Analyze a conversation for signs of contradictions.
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            RutAnalysisResult with the analysis results.
        """
        # Call analyze_conversation to implement the detection logic
        detection_result = self.analyze_conversation(conversation)
        
        # Convert DetectionResult to RutAnalysisResult
        return RutAnalysisResult(
            conversation_id=conversation.id,
            rut_detected=detection_result.rut_detected,
            rut_type=detection_result.rut_type,
            confidence=detection_result.confidence,
            evidence=detection_result.evidence
        )

    def analyze_conversation(self, conversation: Conversation) -> DetectionResult:
        """Analyze a conversation for logical contradictions.
        
        The detector looks for:
        1. Direct contradictions between statements using semantic similarity
        2. Use of polar opposite terms in similar contexts
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            A detection result indicating whether contradictions were detected.
        """
        # Check if there are enough messages
        if len(conversation.messages) < self.min_messages:
            return DetectionResult(
                rut_detected=False,
                rut_type=RutType.CONTRADICTION,
                confidence=0.0,
                evidence={"details": "Not enough messages for contradiction detection"}
            )
            
        # Get recent messages
        recent_messages = conversation.messages[-min(self.window_size, len(conversation.messages)):]
        
        # Extract statements from assistant messages
        assistant_statements = []
        for msg in recent_messages:
            if msg.role == MessageRole.ASSISTANT:
                statements = self._identify_statements(msg.content)
                for statement in statements:
                    assistant_statements.append({
                        "text": statement,
                        "message_idx": conversation.messages.index(msg)
                    })
        
        # If we don't have enough statements, return no contradiction
        if len(assistant_statements) < 2:
            return DetectionResult(
                rut_detected=False,
                rut_type=RutType.CONTRADICTION,
                confidence=0.0,
                evidence={"details": "Not enough statements for contradiction detection"}
            )
            
        # Check for contradictions
        contradictions = []
        
        # Check for direct contradictions using semantic similarity
        direct_contradictions = self._detect_direct_contradictions(assistant_statements)
        if direct_contradictions:
            contradictions.extend(direct_contradictions)
            
        # Check for polar opposite terms
        polar_contradictions = self._detect_polar_opposite_contradictions(assistant_statements)
        if polar_contradictions:
            contradictions.extend(polar_contradictions)
            
        # Calculate confidence based on the number and type of contradictions
        confidence = 0.0
        if contradictions:
            # Use the highest confidence from contradictions found
            confidence = max(c["confidence"] for c in contradictions)
            
        # Create evidence details
        evidence = {
            "contradictions": contradictions,
            "details": f"Found {len(contradictions)} contradictions" if contradictions else "No contradictions detected"
        }
        
        # Determine if contradiction is detected
        rut_detected = confidence >= self.contradiction_threshold
        
        return DetectionResult(
            rut_detected=rut_detected,
            rut_type=RutType.CONTRADICTION,
            confidence=confidence,
            evidence=evidence
        )
    
    def _detect_direct_contradictions(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect direct contradictions between statements using semantic similarity.
        
        Args:
            statements: List of statement objects with text and message index.
            
        Returns:
            List of contradiction objects.
        """
        contradictions = []
        
        # Compare each pair of statements
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                # Get the text of the statements
                text1 = stmt1["text"]
                text2 = stmt2["text"]
                
                # Skip if the statements are too similar (likely repetitions, not contradictions)
                if text1.lower() == text2.lower():
                    continue
                    
                # Get embeddings and calculate similarity
                similarity = self.embeddings_manager.calculate_semantic_similarity(text1, text2)
                
                # Check for negations (indicating potential contradiction)
                negation_indicators = ["not", "n't", "never", "no", "none"]
                has_negation_difference = any(
                    (ind in text1.lower()) != (ind in text2.lower()) 
                    for ind in negation_indicators
                )
                
                # High similarity with negation difference suggests contradiction
                if similarity > 0.8 and has_negation_difference:
                    contradictions.append({
                        "type": "direct_contradiction",
                        "statement1": text1,
                        "statement2": text2,
                        "message_idx1": stmt1["message_idx"],
                        "message_idx2": stmt2["message_idx"],
                        "similarity": similarity,
                        "confidence": similarity * 0.9  # High confidence for direct contradictions
                    })
                    
        return contradictions
    
    def _detect_polar_opposite_contradictions(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions involving polar opposite terms.
        
        Args:
            statements: List of statement objects with text and message index.
            
        Returns:
            List of contradiction objects.
        """
        contradictions = []
        
        # Compare each pair of statements
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                # Get the text of the statements
                text1 = stmt1["text"].lower()
                text2 = stmt2["text"].lower()
                
                # Look for polar opposite terms
                found_opposites = False
                opposite_terms = []
                
                # Check each polar term
                for term, opposites in self.polar_terms.items():
                    if term in text1:
                        for opposite in opposites:
                            if opposite in text2:
                                found_opposites = True
                                opposite_terms.append((term, opposite))
                                
                if found_opposites:
                    # Get context similarity (subtract the opposite terms)
                    text1_filtered = text1
                    text2_filtered = text2
                    for term, opposite in opposite_terms:
                        text1_filtered = text1_filtered.replace(term, "TERM")
                        text2_filtered = text2_filtered.replace(opposite, "TERM")
                        
                    context_similarity = self.embeddings_manager.calculate_semantic_similarity(text1_filtered, text2_filtered)
                    
                    # Only count it as contradiction if the context is similar
                    if context_similarity > 0.7:
                        contradictions.append({
                            "type": "polar_opposite",
                            "statement1": stmt1["text"],
                            "statement2": stmt2["text"],
                            "message_idx1": stmt1["message_idx"],
                            "message_idx2": stmt2["message_idx"],
                            "opposite_terms": opposite_terms,
                            "context_similarity": context_similarity,
                            "confidence": context_similarity * 0.8  # Slightly lower confidence than direct
                        })
                        
        return contradictions 