"""
Utility module for advanced confidence scoring in rut detection.

This module provides functions to calculate, normalize, and aggregate 
confidence scores for various types of rut detections.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import deque

class ConfidenceScorer:
    """
    Provides utilities for advanced confidence scoring in rut detection.
    Computes normalized, weighted, and historically-aware confidence scores.
    """
    
    def __init__(self, 
                history_size: int = 5, 
                smoothing_factor: float = 0.2,
                combine_method: str = "weighted_avg"):
        """
        Initialize the confidence scorer.
        
        Args:
            history_size: Number of previous confidence scores to maintain for each detector
            smoothing_factor: Factor for exponential smoothing (0-1)
            combine_method: Method for combining multiple signals ('weighted_avg', 'max', or 'bayesian')
        """
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        self.combine_method = combine_method
        self.confidence_history = {}  # Maps detector_type -> deque of historical confidences
        
    def normalize_score(self, 
                       raw_score: float, 
                       min_val: float = 0.0, 
                       max_val: float = 1.0,
                       mid_point: float = 0.5,
                       steepness: float = 1.0) -> float:
        """
        Normalize a raw score to the range [0,1] using a sigmoid function.
        
        This applies a sigmoid transformation that can emphasize differences
        around a customizable midpoint.
        
        Args:
            raw_score: The raw confidence score to normalize
            min_val: Theoretical minimum value of the raw score
            max_val: Theoretical maximum value of the raw score
            mid_point: The value that should map to 0.5 confidence
            steepness: Controls the steepness of the sigmoid curve
            
        Returns:
            Normalized confidence score in the range [0,1]
        """
        # Rescale to [0,1] range
        if max_val == min_val:
            scaled_score = 0.5  # Avoid division by zero
        else:
            scaled_score = (raw_score - min_val) / (max_val - min_val)
        
        # Apply sigmoid transformation around midpoint
        # Convert midpoint to the [0,1] scale
        mid_scaled = (mid_point - min_val) / (max_val - min_val)
        
        # Apply sigmoid
        x = steepness * (scaled_score - mid_scaled)
        sigmoid = 1 / (1 + math.exp(-x))
        
        return sigmoid
    
    def combine_signals(self, 
                      confidences: Dict[str, float], 
                      weights: Optional[Dict[str, float]] = None) -> float:
        """
        Combine multiple confidence signals into a single score.
        
        Args:
            confidences: Dictionary mapping signal names to confidence values
            weights: Optional dictionary of weights for each signal
            
        Returns:
            Combined confidence score in the range [0,1]
        """
        if not confidences:
            return 0.0
            
        # Default to equal weights if not provided
        if weights is None:
            weights = {signal: 1.0 / len(confidences) for signal in confidences}
        
        # Ensure all signals have weights
        for signal in confidences:
            if signal not in weights:
                weights[signal] = 0.0
                
        # Normalize weights
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {signal: w / weight_sum for signal, w in weights.items()}
        else:
            weights = {signal: 1.0 / len(confidences) for signal in confidences}
        
        # Apply combination method
        if self.combine_method == "max":
            # Maximum method (highest confidence wins)
            return max(confidences.values()) if confidences else 0.0
            
        elif self.combine_method == "bayesian":
            # Bayesian combination (assuming independent signals)
            # This combines probabilities in a way that increases confidence
            # when multiple signals agree
            prob_no_rut = 1.0
            for signal, conf in confidences.items():
                signal_weight = weights.get(signal, 0.0)
                weighted_conf = conf * signal_weight
                prob_no_rut *= (1.0 - weighted_conf)
            
            return 1.0 - prob_no_rut
            
        else:  # Default to weighted average
            # Weighted average of confidences
            return sum(conf * weights.get(signal, 0.0) 
                     for signal, conf in confidences.items())
    
    def update_history(self, detector_type: str, confidence: float) -> None:
        """
        Update the historical confidence values for a detector.
        
        Args:
            detector_type: The type of detector
            confidence: The current confidence score
        """
        if detector_type not in self.confidence_history:
            self.confidence_history[detector_type] = deque(maxlen=self.history_size)
            
        self.confidence_history[detector_type].append(confidence)
    
    def get_smoothed_confidence(self, detector_type: str, current_confidence: float) -> float:
        """
        Get a smoothed confidence score considering historical values.
        
        Uses exponential smoothing to reduce noise and prevent rapid oscillations.
        
        Args:
            detector_type: The type of detector
            current_confidence: The current raw confidence score
            
        Returns:
            Smoothed confidence score
        """
        # If no history, return current confidence
        if detector_type not in self.confidence_history or not self.confidence_history[detector_type]:
            self.update_history(detector_type, current_confidence)
            return current_confidence
            
        # Get previous smoothed value
        history = list(self.confidence_history[detector_type])
        prev_smoothed = history[-1] if history else current_confidence
        
        # Apply exponential smoothing
        smoothed = (self.smoothing_factor * current_confidence + 
                    (1 - self.smoothing_factor) * prev_smoothed)
        
        # Update history with the new smoothed value
        self.update_history(detector_type, smoothed)
        
        return smoothed
    
    def calculate_trend(self, detector_type: str) -> float:
        """
        Calculate the trend in confidence scores over time.
        
        Returns a value in the range [-1,1] where:
        - Positive values indicate increasing confidence
        - Negative values indicate decreasing confidence
        - Zero indicates stable confidence
        
        Args:
            detector_type: The type of detector
            
        Returns:
            Trend score in the range [-1,1]
        """
        if detector_type not in self.confidence_history:
            return 0.0
            
        history = list(self.confidence_history[detector_type])
        if len(history) < 2:
            return 0.0
            
        # Calculate simple linear trend
        x = np.array(range(len(history)))
        y = np.array(history)
        
        # Use least squares to find slope
        n = len(x)
        if n < 2:
            return 0.0
            
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        
        # Normalize slope to [-1,1]
        # Maximum possible slope would change confidence from 0 to 1 over history_size steps
        max_slope = 1.0 / (len(history) - 1) if len(history) > 1 else 1.0
        normalized_slope = slope / max_slope
        
        # Clamp to [-1,1]
        return max(-1.0, min(1.0, normalized_slope))
    
    def adjust_for_context(self, 
                        confidence: float, 
                        context_factors: Dict[str, float]) -> float:
        """
        Adjust confidence based on contextual factors.
        
        Args:
            confidence: The base confidence score
            context_factors: Dictionary of contextual adjustment factors in the range [-1,1]
            
        Returns:
            Adjusted confidence score
        """
        # Start with base confidence
        adjusted = confidence
        
        # Apply context factors
        for factor, value in context_factors.items():
            # Convert (-1,1) factor to multiplier in range (0.5,1.5)
            multiplier = 1.0 + (value * 0.5)
            adjusted *= multiplier
            
        # Ensure final confidence is in [0,1]
        return max(0.0, min(1.0, adjusted))
    
    def calibrate_threshold(self, 
                          detector_type: str, 
                          desired_sensitivity: float = 0.8,
                          min_threshold: float = 0.3,
                          max_threshold: float = 0.9) -> float:
        """
        Dynamically calibrate detection threshold based on historical performance.
        
        Args:
            detector_type: The type of detector
            desired_sensitivity: Target sensitivity (0-1)
            min_threshold: Minimum allowable threshold
            max_threshold: Maximum allowable threshold
            
        Returns:
            Calibrated threshold value
        """
        # Default to middle threshold if no history
        if detector_type not in self.confidence_history or not self.confidence_history[detector_type]:
            return (min_threshold + max_threshold) / 2
            
        # Get historical confidences
        history = list(self.confidence_history[detector_type])
        
        # Calculate percentile based on desired sensitivity
        # Higher sensitivity -> lower threshold
        percentile = 100 * (1 - desired_sensitivity)
        if history:
            threshold = max(min_threshold, min(max_threshold, np.percentile(history, percentile)))
        else:
            threshold = (min_threshold + max_threshold) / 2
            
        return threshold
    
    def compute_advanced_confidence(self,
                                  detector_type: str,
                                  raw_confidence: float,
                                  signal_weights: Optional[Dict[str, float]] = None,
                                  context_factors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute an advanced confidence score with smoothing, trending, and calibration.
        
        Args:
            detector_type: The type of detector
            raw_confidence: The raw confidence from the detector
            signal_weights: Optional weights for different signals
            context_factors: Optional contextual factors
            
        Returns:
            Dictionary with complete confidence information:
            - raw: The original raw confidence
            - smoothed: Confidence after applying smoothing
            - trend: Trend direction and magnitude (-1 to 1)
            - threshold: Dynamically calibrated threshold
            - final: Final confidence score after all processing
            - is_detected: Boolean indicating if confidence exceeds threshold
        """
        # Apply smoothing
        smoothed = self.get_smoothed_confidence(detector_type, raw_confidence)
        
        # Calculate trend
        trend = self.calculate_trend(detector_type)
        
        # Apply contextual adjustments if provided
        if context_factors:
            smoothed = self.adjust_for_context(smoothed, context_factors)
            
        # Calibrate threshold
        threshold = self.calibrate_threshold(detector_type)
        
        # Combine signals if weights provided
        if signal_weights:
            final = self.combine_signals({detector_type: smoothed}, signal_weights)
        else:
            final = smoothed
            
        # Determine if detection threshold is met
        is_detected = final >= threshold
        
        return {
            "raw": raw_confidence,
            "smoothed": smoothed,
            "trend": trend,
            "threshold": threshold,
            "final": final,
            "is_detected": is_detected
        } 