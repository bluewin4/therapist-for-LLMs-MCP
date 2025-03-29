"""
Intervention strategist for selecting appropriate intervention strategies.

This module provides a strategist that selects appropriate intervention
strategies based on the type of rut detected and the conversation context.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random
import math

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.base import DetectionResult
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    InterventionStrategy,
    RutType
)
from mcp_therapist.utils.logging import logger


class AdaptiveStrategySelector:
    """
    Adaptive selector for intervention strategies based on past effectiveness.
    Uses multi-armed bandit algorithm for exploration-exploitation balance.
    """
    
    def __init__(self, 
                learning_rate: float = 0.1, 
                exploration_factor: float = 0.2,
                algorithm: str = "ucb"):
        """
        Initialize the adaptive strategy selector.
        
        Args:
            learning_rate: Rate at which strategy weights are updated
            exploration_factor: Factor controlling exploration vs. exploitation
            algorithm: Selection algorithm ('epsilon_greedy', 'softmax', or 'ucb')
        """
        self.learning_rate = learning_rate
        self.exploration_factor = exploration_factor
        self.algorithm = algorithm
        
        # Maps rut types to strategy success stats
        # Dict of {rut_type -> {strategy -> {plays, wins, average_reward}}}
        self.strategy_stats = {}
        
        # Default minimum number of plays before considering stats reliable
        self.min_plays = 3
        
    def register_strategy(self, rut_type: RutType, strategy: InterventionStrategy) -> None:
        """
        Register a strategy for a specific rut type.
        
        Args:
            rut_type: The type of rut the strategy addresses
            strategy: The intervention strategy
        """
        if rut_type.value not in self.strategy_stats:
            self.strategy_stats[rut_type.value] = {}
            
        if strategy.value not in self.strategy_stats[rut_type.value]:
            self.strategy_stats[rut_type.value][strategy.value] = {
                "plays": 0,
                "wins": 0,
                "average_reward": 0.5  # Start with neutral success probability
            }
    
    def register_strategies(self, rut_type: RutType, strategies: List[InterventionStrategy]) -> None:
        """
        Register multiple strategies for a specific rut type.
        
        Args:
            rut_type: The type of rut the strategies address
            strategies: List of intervention strategies
        """
        for strategy in strategies:
            self.register_strategy(rut_type, strategy)
    
    def update_stats(self, 
                   rut_type: RutType, 
                   strategy: InterventionStrategy, 
                   was_successful: bool,
                   reward: float = None) -> None:
        """
        Update statistics for a strategy after use.
        
        Args:
            rut_type: The type of rut the strategy addressed
            strategy: The intervention strategy used
            was_successful: Whether the intervention was successful
            reward: Optional specific reward value (0.0 to 1.0)
        """
        if rut_type.value not in self.strategy_stats or strategy.value not in self.strategy_stats[rut_type.value]:
            self.register_strategy(rut_type, strategy)
            
        stats = self.strategy_stats[rut_type.value][strategy.value]
        
        # Increment play count
        stats["plays"] += 1
        
        # Use reward if provided, otherwise 1.0 for success, 0.0 for failure
        if reward is None:
            reward = 1.0 if was_successful else 0.0
            
        # Increment win count if successful
        if was_successful:
            stats["wins"] += 1
            
        # Update average reward using exponential moving average
        old_avg = stats["average_reward"]
        stats["average_reward"] = (1 - self.learning_rate) * old_avg + self.learning_rate * reward
        
        logger.debug(
            f"Updated strategy stats for {strategy.value} ({rut_type.value}): "
            f"plays={stats['plays']}, wins={stats['wins']}, avg_reward={stats['average_reward']:.2f}"
        )
    
    def select_strategy(self, rut_type: RutType, available_strategies: List[InterventionStrategy]) -> InterventionStrategy:
        """
        Select the best strategy based on past performance.
        
        Args:
            rut_type: The type of rut to address
            available_strategies: List of available strategies to choose from
            
        Returns:
            The selected intervention strategy
        """
        # Register strategies if not already tracked
        self.register_strategies(rut_type, available_strategies)
        
        # Filter to strategies that are available
        avail_strat_values = [s.value for s in available_strategies]
        
        # Get stats for available strategies
        if rut_type.value in self.strategy_stats:
            strat_stats = {
                strat: stats
                for strat, stats in self.strategy_stats[rut_type.value].items()
                if strat in avail_strat_values
            }
        else:
            # No stats yet, initialize with all available strategies
            strat_stats = {
                strat.value: {"plays": 0, "wins": 0, "average_reward": 0.5}
                for strat in available_strategies
            }
            
        # Use specified algorithm for selection
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy_selection(rut_type, available_strategies, strat_stats)
        elif self.algorithm == "softmax":
            return self._softmax_selection(rut_type, available_strategies, strat_stats)
        else:  # Default to UCB
            return self._ucb_selection(rut_type, available_strategies, strat_stats)
    
    def _epsilon_greedy_selection(self, 
                              rut_type: RutType, 
                              available_strategies: List[InterventionStrategy],
                              strat_stats: Dict[str, Dict[str, Any]]) -> InterventionStrategy:
        """
        Select strategy using epsilon-greedy algorithm.
        Balances exploitation (choosing best strategy) with exploration (trying others).
        
        Args:
            rut_type: The type of rut to address
            available_strategies: List of available strategies
            strat_stats: Dictionary of strategy statistics
            
        Returns:
            The selected intervention strategy
        """
        # With probability epsilon, explore randomly
        if random.random() < self.exploration_factor:
            # Explore by choosing randomly
            selected_value = random.choice(list(strat_stats.keys()))
            logger.debug(f"Exploration: randomly selected {selected_value} for {rut_type.value}")
        else:
            # Exploit by choosing best strategy
            selected_value = max(strat_stats.items(), key=lambda x: x[1]["average_reward"])[0]
            logger.debug(f"Exploitation: selected best-performing {selected_value} for {rut_type.value}")
            
        # Convert back to InterventionStrategy enum
        return next(s for s in available_strategies if s.value == selected_value)
    
    def _softmax_selection(self, 
                       rut_type: RutType, 
                       available_strategies: List[InterventionStrategy],
                       strat_stats: Dict[str, Dict[str, Any]]) -> InterventionStrategy:
        """
        Select strategy using softmax algorithm.
        Assigns probability proportional to expected reward, with temperature parameter.
        
        Args:
            rut_type: The type of rut to address
            available_strategies: List of available strategies
            strat_stats: Dictionary of strategy statistics
            
        Returns:
            The selected intervention strategy
        """
        # Temperature controls randomness (higher = more exploration)
        temperature = self.exploration_factor
        
        # Calculate selection probabilities using softmax
        rewards = [strat_stats[s.value]["average_reward"] for s in available_strategies]
        
        # To avoid overflow, subtract max reward before exponentiation
        max_reward = max(rewards)
        exp_rewards = [math.exp((r - max_reward) / temperature) for r in rewards]
        total_exp_reward = sum(exp_rewards)
        
        # Calculate probabilities
        probabilities = [er / total_exp_reward for er in exp_rewards]
        
        # Select strategy based on calculated probabilities
        selected_index = random.choices(range(len(available_strategies)), weights=probabilities)[0]
        selected_strategy = available_strategies[selected_index]
        
        logger.debug(
            f"Softmax selection: chose {selected_strategy.value} for {rut_type.value} "
            f"(probability: {probabilities[selected_index]:.2f})"
        )
        
        return selected_strategy
    
    def _ucb_selection(self, 
                   rut_type: RutType, 
                   available_strategies: List[InterventionStrategy],
                   strat_stats: Dict[str, Dict[str, Any]]) -> InterventionStrategy:
        """
        Select strategy using Upper Confidence Bound (UCB) algorithm.
        Balances exploitation with exploration based on uncertainty.
        
        Args:
            rut_type: The type of rut to address
            available_strategies: List of available strategies
            strat_stats: Dictionary of strategy statistics
            
        Returns:
            The selected intervention strategy
        """
        # Total number of plays across all strategies
        total_plays = sum(stats["plays"] for stats in strat_stats.values())
        
        # If no plays yet, select randomly
        if total_plays == 0:
            selected_strategy = random.choice(available_strategies)
            logger.debug(f"No plays yet, randomly selected {selected_strategy.value} for {rut_type.value}")
            return selected_strategy
            
        # Calculate UCB scores for each strategy
        ucb_scores = {}
        
        for strat_value, stats in strat_stats.items():
            # Exploitation component
            exploit = stats["average_reward"]
            
            # Exploration component - increases with total plays and decreases with strategy plays
            if stats["plays"] == 0:
                # If strategy never played, give it high exploration value
                explore = 1.0
            else:
                # UCB formula with exploration factor as constant C
                explore = self.exploration_factor * math.sqrt(2 * math.log(total_plays) / stats["plays"])
                
            # Calculate UCB score
            ucb_scores[strat_value] = exploit + explore
            
        # Select strategy with highest UCB score
        selected_value = max(ucb_scores.items(), key=lambda x: x[1])[0]
        logger.debug(
            f"UCB selection: chose {selected_value} for {rut_type.value} "
            f"(exploit: {strat_stats[selected_value]['average_reward']:.2f}, "
            f"explore: {ucb_scores[selected_value] - strat_stats[selected_value]['average_reward']:.2f})"
        )
        
        # Convert back to InterventionStrategy enum
        return next(s for s in available_strategies if s.value == selected_value)


class InterventionStrategist:
    """Strategist for selecting appropriate intervention strategies."""
    
    def __init__(self):
        """Initialize the intervention strategist."""
        self.logger = logger
        
        # Configure cooldown settings
        self.intervention_cooldown = settings.INTERVENTION_COOLDOWN
        self.max_interventions = settings.MAX_INTERVENTIONS_PER_CONVERSATION
        
        # Strategy mappings: map rut types to prioritized lists of strategies
        self.strategy_mapping = {
            RutType.REPETITION: [
                InterventionStrategy.REFLECTION,
                InterventionStrategy.PROMPT_REFINEMENT
            ],
            RutType.STAGNATION: [
                InterventionStrategy.REFRAMING,
                InterventionStrategy.TOPIC_SWITCH,
                InterventionStrategy.EXPLORATION
            ],
            RutType.REFUSAL: [
                InterventionStrategy.CLARIFY_CONSTRAINTS,
                InterventionStrategy.REFRAME_REQUEST,
                InterventionStrategy.EXPLORATION
            ],
            RutType.NEGATIVITY: [
                InterventionStrategy.POSITIVE_REFRAMING,
                InterventionStrategy.REFLECTION,
                InterventionStrategy.TOPIC_SWITCH
            ],
            RutType.CONTRADICTION: [
                InterventionStrategy.HIGHLIGHT_INCONSISTENCY,
                InterventionStrategy.REQUEST_CLARIFICATION
            ],
            RutType.TOPIC_FIXATION: [
                InterventionStrategy.BROADEN_TOPIC,
                InterventionStrategy.TOPIC_SWITCH,
                InterventionStrategy.EXPLORATION
            ]
        }
        
        # Initialize adaptive strategy selector with settings from config
        self.adaptive_selector = AdaptiveStrategySelector(
            learning_rate=settings.STRATEGY_LEARNING_RATE,
            exploration_factor=settings.STRATEGY_EXPLORATION_FACTOR,
            algorithm=settings.STRATEGY_SELECTION_ALGORITHM
        )
        
        # Register all strategies with the adaptive selector
        for rut_type, strategies in self.strategy_mapping.items():
            self.adaptive_selector.register_strategies(rut_type, strategies)
        
        # Legacy strategy effectiveness tracking
        # Maps strategies to success rates
        self.strategy_success_rates: Dict[InterventionStrategy, float] = {
            strategy: 0.5  # Initialize with neutral success rate
            for strategy in InterventionStrategy
        }
    
    def should_intervene(self, conversation: Conversation, detection_result: DetectionResult) -> bool:
        """Determine whether an intervention should be made.
        
        Args:
            conversation: The conversation to potentially intervene in.
            detection_result: The result of the rut detection.
            
        Returns:
            True if an intervention should be made, False otherwise.
        """
        # Don't intervene if no rut was detected
        if not detection_result.rut_detected:
            return False
        
        # Check if we've reached the maximum number of interventions
        intervention_count = sum(1 for msg in conversation.messages if msg.metadata.get("is_intervention", False))
        if intervention_count >= self.max_interventions:
            self.logger.info(
                f"Maximum interventions ({self.max_interventions}) reached for conversation {conversation.id}"
            )
            return False
        
        # Check for cooldown period
        last_intervention = None
        for i, msg in enumerate(reversed(conversation.messages)):
            if msg.metadata.get("is_intervention", False):
                last_intervention = msg
                break
                
        if last_intervention:
            # Calculate messages since last intervention
            last_idx = conversation.messages.index(last_intervention)
            messages_since = len(conversation.messages) - last_idx - 1
            
            if messages_since < self.intervention_cooldown:
                self.logger.info(
                    f"Cooldown active: {messages_since}/{self.intervention_cooldown} messages since last intervention"
                )
                return False
        
        # Consider confidence level in detection result
        min_confidence = getattr(settings, "MIN_INTERVENTION_CONFIDENCE", 0.7)
        if detection_result.confidence < min_confidence:
            self.logger.info(
                f"Confidence too low for intervention: {detection_result.confidence:.2f} < "
                f"{min_confidence}"
            )
            return False
        
        return True
    
    def select_strategy(self, detection_result: DetectionResult, conversation: Conversation) -> InterventionStrategy:
        """Select a strategy based on the rut type and conversation context.
        
        Args:
            detection_result: The result of the rut detection.
            conversation: The conversation context.
            
        Returns:
            The selected intervention strategy.
        """
        rut_type = detection_result.rut_type
        
        # Get the strategy options for this rut type
        strategies = self.strategy_mapping.get(rut_type, [InterventionStrategy.OTHER])
        
        # Check if we've tried any of these strategies recently
        recent_intervention_msgs = []
        for msg in reversed(conversation.messages):
            if msg.metadata.get("is_intervention", False):
                recent_intervention_msgs.append(msg)
                if len(recent_intervention_msgs) >= 3:
                    break
                    
        recent_strategies = [
            InterventionStrategy(msg.metadata.get("strategy_type", "OTHER"))
            for msg in recent_intervention_msgs
            if "strategy_type" in msg.metadata
        ]
        
        # Filter out recently used strategies, unless they're all we have
        available_strategies = [
            s for s in strategies 
            if s not in recent_strategies or len(strategies) <= len(recent_strategies)
        ]
        
        if not available_strategies:
            available_strategies = strategies
        
        # Use adaptive strategy selection
        selected_strategy = self.adaptive_selector.select_strategy(rut_type, available_strategies)
        
        self.logger.info(
            f"Selected strategy {selected_strategy.value} for {rut_type.value} rut "
            f"(adaptive selection)"
        )
        
        return selected_strategy
    
    def create_intervention_plan(
        self, detection_result: DetectionResult, conversation: Conversation
    ) -> Optional[InterventionPlan]:
        """Create an intervention plan based on the detection result.
        
        Args:
            detection_result: The result of the rut detection.
            conversation: The conversation context.
            
        Returns:
            An intervention plan, or None if no intervention should be made.
        """
        # Check if we should intervene
        if not self.should_intervene(conversation, detection_result):
            return None
        
        # Select a strategy
        strategy = self.select_strategy(detection_result, conversation)
        
        # Extract target information and context from detection
        context_info = self._extract_context(detection_result, conversation)
        
        # Create the intervention plan
        plan = InterventionPlan(
            conversation_id=conversation.id,
            rut_type=detection_result.rut_type,
            strategy_type=strategy.value,
            confidence=detection_result.confidence,
            metadata={
                "rut_type": detection_result.rut_type.value,
                "confidence": detection_result.confidence,
                "created_at": datetime.now().isoformat(),
                "detection_evidence": detection_result.evidence,
                "context": context_info
            }
        )
        
        self.logger.info(
            f"Created intervention plan with strategy {strategy.value} "
            f"for {detection_result.rut_type.value} rut"
        )
        
        return plan
    
    def _extract_context(self, detection_result: DetectionResult, conversation: Conversation) -> Dict[str, Any]:
        """Extract contextual information for the intervention.
        
        Args:
            detection_result: The detection result
            conversation: The conversation context
            
        Returns:
            Dictionary with context information
        """
        context = {}
        
        # Extract target topic from the detection result
        if detection_result.rut_type == RutType.TOPIC_FIXATION and "repeated_terms" in detection_result.evidence:
            repeated_terms = detection_result.evidence["repeated_terms"]
            if repeated_terms:
                context["target_topic"] = " ".join(repeated_terms[:3])
                
        # Extract contradictions for the contradiction detector
        if detection_result.rut_type == RutType.CONTRADICTION and "contradictions" in detection_result.evidence:
            contradictions = detection_result.evidence["contradictions"]
            if contradictions:
                context["contradictions"] = contradictions
                
        # Extract conversation stats
        context["message_count"] = len(conversation.messages)
        context["user_message_count"] = sum(1 for m in conversation.messages if m.role == MessageRole.USER)
        context["assistant_message_count"] = sum(1 for m in conversation.messages if m.role == MessageRole.ASSISTANT)
        
        # Extract interaction history
        intervention_count = sum(1 for m in conversation.messages if m.metadata.get("is_intervention", False))
        context["intervention_count"] = intervention_count
        
        return context
    
    def update_strategy_effectiveness(
        self, strategy: InterventionStrategy, rut_type: RutType, was_successful: bool, reward: float = None
    ) -> None:
        """Update the effectiveness tracking for a strategy.
        
        Args:
            strategy: The strategy to update.
            rut_type: The type of rut that was addressed.
            was_successful: Whether the intervention was successful.
            reward: Optional specific reward value (0.0 to 1.0).
        """
        # Legacy update
        current_rate = self.strategy_success_rates.get(strategy, 0.5)
        
        # Update success rate using weighted moving average
        if was_successful:
            new_rate = current_rate * 0.9 + 1.0 * 0.1
        else:
            new_rate = current_rate * 0.9 + 0.0 * 0.1
        
        # Update the tracking
        self.strategy_success_rates[strategy] = new_rate
        
        # Update adaptive selector
        self.adaptive_selector.update_stats(rut_type, strategy, was_successful, reward)
        
        self.logger.debug(
            f"Updated success rate for {strategy.value}: {current_rate:.2f} -> {new_rate:.2f} "
            f"(was_successful={was_successful})"
        ) 