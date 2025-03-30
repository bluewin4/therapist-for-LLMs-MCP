"""Pipeline for running multiple detectors on conversations."""

import time
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

from mcp_therapist.core.detectors.base import Detector, DetectionResult
from mcp_therapist.core.conversation import Conversation, Message
from mcp_therapist.config import settings
from mcp_therapist.utils import logging
from mcp_therapist.utils.caching import memoize
from mcp_therapist.utils.concurrency import task_manager
from mcp_therapist.utils.profiling import Profiler

logger = logging.get_logger(__name__)


class DetectorPipeline:
    """Pipeline for running multiple detectors on a conversation."""

    def __init__(self, detectors: Optional[List[Detector]] = None):
        """
        Initialize a detector pipeline.

        Args:
            detectors: List of detectors to run (if None, an empty list is used)
        """
        self.detectors = detectors or []
        self.profiler = Profiler.get_instance()
    
    def add_detector(self, detector: Detector) -> None:
        """
        Add a detector to the pipeline.

        Args:
            detector: Detector to add
        """
        self.detectors.append(detector)

    @memoize(ttl=60)  # Cache results for 1 minute
    def detect(self, conversation: Conversation) -> Dict[str, DetectionResult]:
        """
        Run all detectors on a conversation.

        Args:
            conversation: Conversation to analyze

        Returns:
            Dict mapping detector IDs to detection results
        """
        with self.profiler.profile_section("detector_pipeline.detect"):
            if not self.detectors:
                return {}
            
            # Check if we should use parallel processing
            if settings.performance.parallel_detection and len(self.detectors) > 1:
                return self._detect_parallel(conversation)
            else:
                return self._detect_sequential(conversation)
    
    def _detect_sequential(self, conversation: Conversation) -> Dict[str, DetectionResult]:
        """
        Run all detectors sequentially.

        Args:
            conversation: Conversation to analyze

        Returns:
            Dict mapping detector IDs to detection results
        """
        results = {}
        
        for detector in self.detectors:
            detector_id = detector.id
            
            with self.profiler.profile_section(f"detector.{detector_id}"):
                try:
                    result = detector.detect(conversation)
                    results[detector_id] = result
                except Exception as e:
                    logger.error(f"Error in detector {detector_id}: {e}")
                    # Create an empty result
                    results[detector_id] = DetectionResult(
                        detected=False,
                        confidence=0.0,
                        metadata={
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
        
        return results
    
    def _detect_parallel(self, conversation: Conversation) -> Dict[str, DetectionResult]:
        """
        Run all detectors in parallel.

        Args:
            conversation: Conversation to analyze

        Returns:
            Dict mapping detector IDs to detection results
        """
        detector_ids = [detector.id for detector in self.detectors]
        
        # Define function to run a single detector
        def run_detector(detector: Detector) -> Tuple[str, DetectionResult]:
            detector_id = detector.id
            with self.profiler.profile_section(f"detector.{detector_id}"):
                try:
                    result = detector.detect(conversation)
                    return detector_id, result
                except Exception as e:
                    logger.error(f"Error in detector {detector_id}: {e}")
                    # Create an empty result
                    return detector_id, DetectionResult(
                        detected=False,
                        confidence=0.0,
                        metadata={
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
        
        # Run detectors in batches
        max_batch_size = settings.performance.detector_batch_size or 5
        
        # Submit detector tasks in batches
        results_list = task_manager.run_in_batch(
            run_detector,
            self.detectors,
            max_batch_size=max_batch_size
        )
        
        # Convert list of tuples to dict
        results = {}
        for detector_id, result in results_list:
            if detector_id and result:
                results[detector_id] = result
        
        return results
    
    def detect_incremental(
        self, 
        conversation: Conversation,
        last_processed_index: Optional[int] = None
    ) -> Dict[str, DetectionResult]:
        """
        Run detectors on new messages in a conversation.
        
        This optimizes processing by only analyzing new messages
        since the last analysis.
        
        Args:
            conversation: Conversation to analyze
            last_processed_index: Index of the last processed message
            
        Returns:
            Dict mapping detector IDs to detection results
        """
        # If no previous processing or only 1-2 messages, process the whole conversation
        if (
            last_processed_index is None or 
            last_processed_index < 0 or
            len(conversation.messages) <= 2 or
            len(conversation.messages) - last_processed_index <= 2
        ):
            return self.detect(conversation)
        
        # Create a slice of the conversation with context
        # Include 2 messages before the new ones for context
        context_start = max(0, last_processed_index - 1)
        conversation_slice = Conversation(
            messages=conversation.messages[context_start:],
            id=conversation.id,
            metadata=conversation.metadata.copy()
        )
        
        # Add metadata to indicate this is incremental
        conversation_slice.metadata["incremental_processing"] = True
        conversation_slice.metadata["full_conversation_length"] = len(conversation.messages)
        conversation_slice.metadata["slice_start_index"] = context_start
        
        # Detect on the slice
        return self.detect(conversation_slice)
    
    def get_batch_results(
        self,
        conversations: List[Conversation]
    ) -> List[Dict[str, DetectionResult]]:
        """
        Process multiple conversations in batch mode.
        
        Args:
            conversations: List of conversations to analyze
            
        Returns:
            List of results dictionaries, one per conversation
        """
        if not conversations:
            return []
        
        # If parallel processing is enabled and we have multiple conversations
        if settings.performance.parallel_batch_processing and len(conversations) > 1:
            # Define function to process a single conversation
            def process_conversation(conv: Conversation) -> Dict[str, DetectionResult]:
                return self.detect(conv)
            
            # Process conversations in parallel batches
            max_batch_size = settings.performance.conversation_batch_size or 3
            return task_manager.run_in_batch(
                process_conversation,
                conversations,
                max_batch_size=max_batch_size
            )
        else:
            # Process sequentially
            results = []
            for conv in conversations:
                results.append(self.detect(conv))
            return results
    
    def analyzer_generator(
        self, 
        conversation: Conversation,
        interval: float = 0.5
    ) -> Iterator[Dict[str, DetectionResult]]:
        """
        Generate detection results at regular intervals for streaming.
        
        This is useful for analyzing a conversation in real-time
        as new messages arrive.
        
        Args:
            conversation: Conversation to analyze
            interval: Time between analyses in seconds
            
        Yields:
            Dict mapping detector IDs to detection results
        """
        last_message_count = 0
        last_processed_index = -1
        
        while True:
            current_count = len(conversation.messages)
            
            # If there are new messages, analyze them
            if current_count > last_message_count:
                results = self.detect_incremental(
                    conversation,
                    last_processed_index=last_processed_index
                )
                last_message_count = current_count
                last_processed_index = current_count - 1
                
                yield results
            
            # Wait for the next interval
            time.sleep(interval) 