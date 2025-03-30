"""
Threshold optimization module for MCP Therapist.

This module provides tools for automatically optimizing detection and
intervention thresholds based on performance metrics and feedback.
"""

import math
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import os
from pathlib import Path

from mcp_therapist.config import settings
from mcp_therapist.utils import logging
from mcp_therapist.utils.profiling import Profiler
from mcp_therapist.core.conversation import Conversation
from mcp_therapist.core.detectors.base import DetectionResult

logger = logging.get_logger(__name__)


@dataclass
class OptimizationResult:
    """Results of a threshold optimization round."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "improvement": self.improvement
        }


@dataclass
class ThresholdOptimizerConfig:
    """Configuration for threshold optimization."""
    # Optimization parameters
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    step_size: float = 0.05
    target_metric: str = "f1_score"  # "precision", "recall", "f1_score"
    
    # Convergence parameters
    min_improvement: float = 0.01
    max_iterations: int = 10
    
    # Weighting for false positives/negatives
    false_positive_weight: float = 1.0
    false_negative_weight: float = 1.0
    
    # Learning rate for online updates
    learning_rate: float = 0.05
    
    # Minimum samples required for optimization
    min_samples: int = 50


class ThresholdOptimizer:
    """
    Optimizer for detection and intervention thresholds.
    
    This class provides methods for automatically adjusting
    confidence thresholds based on performance metrics.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[ThresholdOptimizerConfig] = None,
        initial_threshold: Optional[float] = None
    ):
        """
        Initialize the threshold optimizer.
        
        Args:
            name: Name of the component being optimized
            config: Optimizer configuration
            initial_threshold: Initial threshold value
        """
        self.name = name
        self.config = config or ThresholdOptimizerConfig()
        self.threshold = initial_threshold or 0.7
        
        # Performance tracking
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # History tracking
        self.history: List[OptimizationResult] = []
        
        # Profiler
        self.profiler = Profiler.get_instance()
        
        # Load history if available
        self._load_history()
    
    def record_result(
        self, 
        predicted: bool, 
        actual: bool, 
        confidence: Optional[float] = None
    ) -> None:
        """
        Record a prediction result.
        
        Args:
            predicted: The predicted value (True for detected, False for not detected)
            actual: The actual value (True for should have detected, False for shouldn't)
            confidence: The confidence score for the prediction, if available
        """
        # Update confusion matrix
        if predicted and actual:
            self.true_positives += 1
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and actual:
            self.false_negatives += 1
        else:  # not predicted and not actual
            self.true_negatives += 1
        
        # If confidence available, do online update of threshold
        if confidence is not None and settings.performance.enable_threshold_auto_update:
            self._update_threshold_online(predicted, actual, confidence)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate metrics
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        f1 = self._calculate_f1_score()
        fpr = self._calculate_false_positive_rate()
        fnr = self._calculate_false_negative_rate()
        
        # Return metrics
        return {
            "threshold": self.threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "sample_count": self.true_positives + self.false_positives + 
                            self.true_negatives + self.false_negatives
        }
    
    def optimize_threshold(
        self,
        predictions: List[Tuple[float, bool]],
        save_result: bool = True
    ) -> OptimizationResult:
        """
        Optimize threshold based on a list of predictions.
        
        Args:
            predictions: List of (confidence, actual) tuples
            save_result: Whether to save the result to history
            
        Returns:
            Optimization result
        """
        with self.profiler.profile_section(f"threshold_optimizer.{self.name}"):
            # Check if we have enough samples
            if len(predictions) < self.config.min_samples:
                logger.warning(
                    f"Not enough samples to optimize threshold for {self.name}: "
                    f"{len(predictions)} < {self.config.min_samples}"
                )
                return OptimizationResult(
                    threshold=self.threshold,
                    precision=self._calculate_precision(),
                    recall=self._calculate_recall(),
                    f1_score=self._calculate_f1_score(),
                    false_positive_rate=self._calculate_false_positive_rate(),
                    false_negative_rate=self._calculate_false_negative_rate(),
                    improvement=0.0
                )
            
            # Sort predictions by confidence
            predictions.sort(key=lambda x: x[0])
            
            # Get current best metric
            current_metric = self._get_target_metric()
            
            # Try different thresholds
            best_threshold = self.threshold
            best_metric = current_metric
            best_result = None
            
            # Test thresholds
            for threshold in np.arange(
                self.config.min_threshold,
                self.config.max_threshold + self.config.step_size,
                self.config.step_size
            ):
                # Apply threshold
                tp, fp, tn, fn = self._apply_threshold(predictions, threshold)
                
                # Calculate metrics
                precision = self._calculate_precision_raw(tp, fp)
                recall = self._calculate_recall_raw(tp, fn)
                f1 = self._calculate_f1_score_raw(precision, recall)
                fpr = self._calculate_false_positive_rate_raw(fp, tn)
                fnr = self._calculate_false_negative_rate_raw(fn, tp)
                
                # Get metric based on target
                if self.config.target_metric == "precision":
                    metric = precision
                elif self.config.target_metric == "recall":
                    metric = recall
                else:  # f1_score
                    metric = f1
                
                # Check if this is an improvement
                if metric > best_metric:
                    best_threshold = threshold
                    best_metric = metric
                    best_result = OptimizationResult(
                        threshold=threshold,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        false_positive_rate=fpr,
                        false_negative_rate=fnr,
                        improvement=metric - current_metric
                    )
            
            # If we found an improvement, update threshold
            if best_result and best_result.improvement > self.config.min_improvement:
                self.threshold = best_threshold
                logger.info(
                    f"Optimized threshold for {self.name}: "
                    f"{self.threshold:.4f} -> {best_threshold:.4f} "
                    f"(improvement: {best_result.improvement:.4f})"
                )
                
                # Save to history
                if save_result:
                    self.history.append(best_result)
                    self._save_history()
                
                return best_result
            else:
                # No significant improvement
                logger.info(
                    f"No significant threshold improvement for {self.name}. "
                    f"Keeping current threshold: {self.threshold:.4f}"
                )
                return OptimizationResult(
                    threshold=self.threshold,
                    precision=self._calculate_precision(),
                    recall=self._calculate_recall(),
                    f1_score=self._calculate_f1_score(),
                    false_positive_rate=self._calculate_false_positive_rate(),
                    false_negative_rate=self._calculate_false_negative_rate(),
                    improvement=0.0
                )
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def _update_threshold_online(
        self, 
        predicted: bool, 
        actual: bool, 
        confidence: float
    ) -> None:
        """
        Update threshold based on a single prediction.
        
        Args:
            predicted: Whether the prediction was positive
            actual: Whether the actual value was positive
            confidence: Confidence score
        """
        # Adjust threshold based on error type
        if predicted and not actual:  # False positive
            # Increase threshold to reduce false positives
            adjustment = self.config.false_positive_weight * self.config.learning_rate
            new_threshold = min(
                self.threshold + adjustment,
                self.config.max_threshold
            )
            self.threshold = new_threshold
        elif not predicted and actual:  # False negative
            # Decrease threshold to reduce false negatives
            adjustment = self.config.false_negative_weight * self.config.learning_rate
            new_threshold = max(
                self.threshold - adjustment,
                self.config.min_threshold
            )
            self.threshold = new_threshold
    
    def _apply_threshold(
        self, 
        predictions: List[Tuple[float, bool]], 
        threshold: float
    ) -> Tuple[int, int, int, int]:
        """
        Apply a threshold to predictions and calculate confusion matrix.
        
        Args:
            predictions: List of (confidence, actual) tuples
            threshold: Threshold to apply
            
        Returns:
            Tuple of (true_positives, false_positives, true_negatives, false_negatives)
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for confidence, actual in predictions:
            predicted = confidence >= threshold
            
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:  # not predicted and not actual
                tn += 1
        
        return tp, fp, tn, fn
    
    def _calculate_precision(self) -> float:
        """
        Calculate precision from current stats.
        
        Returns:
            Precision value
        """
        return self._calculate_precision_raw(self.true_positives, self.false_positives)
    
    def _calculate_precision_raw(self, tp: int, fp: int) -> float:
        """
        Calculate precision from raw values.
        
        Args:
            tp: True positives
            fp: False positives
            
        Returns:
            Precision value
        """
        if tp + fp == 0:
            return 1.0  # Avoid division by zero
        return tp / (tp + fp)
    
    def _calculate_recall(self) -> float:
        """
        Calculate recall from current stats.
        
        Returns:
            Recall value
        """
        return self._calculate_recall_raw(self.true_positives, self.false_negatives)
    
    def _calculate_recall_raw(self, tp: int, fn: int) -> float:
        """
        Calculate recall from raw values.
        
        Args:
            tp: True positives
            fn: False negatives
            
        Returns:
            Recall value
        """
        if tp + fn == 0:
            return 1.0  # Avoid division by zero
        return tp / (tp + fn)
    
    def _calculate_f1_score(self) -> float:
        """
        Calculate F1 score from current stats.
        
        Returns:
            F1 score value
        """
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        return self._calculate_f1_score_raw(precision, recall)
    
    def _calculate_f1_score_raw(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score value
        """
        if precision + recall == 0:
            return 0.0  # Avoid division by zero
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_false_positive_rate(self) -> float:
        """
        Calculate false positive rate from current stats.
        
        Returns:
            False positive rate
        """
        return self._calculate_false_positive_rate_raw(
            self.false_positives, self.true_negatives
        )
    
    def _calculate_false_positive_rate_raw(self, fp: int, tn: int) -> float:
        """
        Calculate false positive rate from raw values.
        
        Args:
            fp: False positives
            tn: True negatives
            
        Returns:
            False positive rate
        """
        if fp + tn == 0:
            return 0.0  # Avoid division by zero
        return fp / (fp + tn)
    
    def _calculate_false_negative_rate(self) -> float:
        """
        Calculate false negative rate from current stats.
        
        Returns:
            False negative rate
        """
        return self._calculate_false_negative_rate_raw(
            self.false_negatives, self.true_positives
        )
    
    def _calculate_false_negative_rate_raw(self, fn: int, tp: int) -> float:
        """
        Calculate false negative rate from raw values.
        
        Args:
            fn: False negatives
            tp: True positives
            
        Returns:
            False negative rate
        """
        if fn + tp == 0:
            return 0.0  # Avoid division by zero
        return fn / (fn + tp)
    
    def _get_target_metric(self) -> float:
        """
        Get current value of target metric.
        
        Returns:
            Current metric value
        """
        if self.config.target_metric == "precision":
            return self._calculate_precision()
        elif self.config.target_metric == "recall":
            return self._calculate_recall()
        else:  # f1_score
            return self._calculate_f1_score()
    
    def _get_history_path(self) -> Path:
        """
        Get path for history file.
        
        Returns:
            Path to history file
        """
        # Create optimization directory if it doesn't exist
        base_dir = Path(settings.optimization_dir) if hasattr(settings, "optimization_dir") else Path("optimization")
        os.makedirs(base_dir, exist_ok=True)
        
        # Return path to history file
        return base_dir / f"{self.name}_threshold_history.json"
    
    def _save_history(self) -> None:
        """Save optimization history to file."""
        try:
            # Convert history to list of dictionaries
            history_dicts = [result.to_dict() for result in self.history]
            
            # Save to file
            with open(self._get_history_path(), "w") as f:
                json.dump({
                    "name": self.name,
                    "threshold": self.threshold,
                    "history": history_dicts
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving threshold history for {self.name}: {e}")
    
    def _load_history(self) -> None:
        """Load optimization history from file."""
        try:
            history_path = self._get_history_path()
            if history_path.exists():
                with open(history_path, "r") as f:
                    data = json.load(f)
                
                # Load threshold
                self.threshold = data.get("threshold", self.threshold)
                
                # Load history
                self.history = []
                for item in data.get("history", []):
                    self.history.append(OptimizationResult(
                        threshold=item["threshold"],
                        precision=item["precision"],
                        recall=item["recall"],
                        f1_score=item["f1_score"],
                        false_positive_rate=item["false_positive_rate"],
                        false_negative_rate=item["false_negative_rate"],
                        improvement=item["improvement"]
                    ))
        except Exception as e:
            logger.error(f"Error loading threshold history for {self.name}: {e}")


class ThresholdRegistry:
    """
    Registry for managing multiple threshold optimizers.
    
    This class provides a centralized way to access and manage
    threshold optimizers for different components.
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ThresholdRegistry':
        """
        Get the singleton instance of the registry.
        
        Returns:
            ThresholdRegistry instance
        """
        if cls._instance is None:
            cls._instance = ThresholdRegistry()
        return cls._instance
    
    def __init__(self):
        """Initialize the threshold registry."""
        self.optimizers: Dict[str, ThresholdOptimizer] = {}
    
    def get_optimizer(
        self,
        name: str,
        config: Optional[ThresholdOptimizerConfig] = None,
        initial_threshold: Optional[float] = None
    ) -> ThresholdOptimizer:
        """
        Get or create an optimizer for a component.
        
        Args:
            name: Name of the component
            config: Optimizer configuration
            initial_threshold: Initial threshold value
            
        Returns:
            ThresholdOptimizer instance
        """
        if name not in self.optimizers:
            self.optimizers[name] = ThresholdOptimizer(
                name=name,
                config=config,
                initial_threshold=initial_threshold
            )
        return self.optimizers[name]
    
    def get_presets(self) -> Dict[str, float]:
        """
        Get current threshold presets for all components.
        
        Returns:
            Dictionary mapping component names to threshold values
        """
        return {name: optimizer.threshold for name, optimizer in self.optimizers.items()}
    
    def optimize_all(self) -> Dict[str, OptimizationResult]:
        """
        Run optimization for all registered optimizers.
        
        Returns:
            Dictionary mapping component names to optimization results
        """
        results = {}
        for name, optimizer in self.optimizers.items():
            # Skip if not enough data
            total_samples = (
                optimizer.true_positives + optimizer.false_positives +
                optimizer.true_negatives + optimizer.false_negatives
            )
            if total_samples < optimizer.config.min_samples:
                logger.info(
                    f"Skipping optimization for {name}: "
                    f"not enough samples ({total_samples})"
                )
                continue
            
            # Create prediction list
            predictions = []
            
            # We don't have actual confidence values in the stats,
            # so we simulate them based on the confusion matrix
            
            # True positives (confidence > threshold)
            tp_confidences = np.linspace(
                optimizer.threshold,
                1.0,
                optimizer.true_positives
            )
            for conf in tp_confidences:
                predictions.append((conf, True))
            
            # False positives (confidence > threshold)
            fp_confidences = np.linspace(
                optimizer.threshold,
                1.0,
                optimizer.false_positives
            )
            for conf in fp_confidences:
                predictions.append((conf, False))
            
            # True negatives (confidence < threshold)
            tn_confidences = np.linspace(
                0.0,
                optimizer.threshold - 0.001,
                optimizer.true_negatives
            )
            for conf in tn_confidences:
                predictions.append((conf, False))
            
            # False negatives (confidence < threshold)
            fn_confidences = np.linspace(
                0.0,
                optimizer.threshold - 0.001,
                optimizer.false_negatives
            )
            for conf in fn_confidences:
                predictions.append((conf, True))
            
            # Run optimization
            results[name] = optimizer.optimize_threshold(predictions)
            
            # Reset stats
            optimizer.reset_stats()
        
        return results


# Create global registry
threshold_registry = ThresholdRegistry.get_instance() 