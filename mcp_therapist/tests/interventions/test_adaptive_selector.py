"""Tests for the AdaptiveStrategySelector class."""

import unittest
from unittest.mock import patch, MagicMock
import math
import random

from mcp_therapist.core.interventions.strategist import AdaptiveStrategySelector
from mcp_therapist.models.conversation import RutType, InterventionStrategy


class TestAdaptiveStrategySelector(unittest.TestCase):
    """Test cases for the AdaptiveStrategySelector class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Use a fixed seed for reproducible tests
        random.seed(42)
        
        # Create a selector with test parameters
        self.selector = AdaptiveStrategySelector(
            learning_rate=0.2,
            exploration_factor=0.3,
            algorithm="ucb"
        )
        
        # Set up test strategies and rut types
        self.rut_type = RutType.STAGNATION
        self.strategies = [
            InterventionStrategy.REFRAMING,
            InterventionStrategy.TOPIC_SWITCH,
            InterventionStrategy.EXPLORATION
        ]
        
        # Register strategies with the selector
        self.selector.register_strategies(self.rut_type, self.strategies)
    
    def test_initialization(self):
        """Test the initialization of the selector."""
        self.assertEqual(self.selector.learning_rate, 0.2)
        self.assertEqual(self.selector.exploration_factor, 0.3)
        self.assertEqual(self.selector.algorithm, "ucb")
        self.assertEqual(self.selector.min_plays, 3)
        self.assertTrue(isinstance(self.selector.strategy_stats, dict))
    
    def test_register_strategy(self):
        """Test registering a single strategy."""
        test_rut = RutType.REPETITION
        test_strategy = InterventionStrategy.REFLECTION
        
        self.selector.register_strategy(test_rut, test_strategy)
        
        # Check strategy was registered
        self.assertIn(test_rut.value, self.selector.strategy_stats)
        self.assertIn(test_strategy.value, self.selector.strategy_stats[test_rut.value])
        
        # Check default values
        stats = self.selector.strategy_stats[test_rut.value][test_strategy.value]
        self.assertEqual(stats["plays"], 0)
        self.assertEqual(stats["wins"], 0)
        self.assertEqual(stats["average_reward"], 0.5)
    
    def test_register_strategies(self):
        """Test registering multiple strategies at once."""
        test_rut = RutType.REFUSAL
        test_strategies = [
            InterventionStrategy.CLARIFY_CONSTRAINTS, 
            InterventionStrategy.REFRAME_REQUEST
        ]
        
        self.selector.register_strategies(test_rut, test_strategies)
        
        # Check all strategies were registered
        self.assertIn(test_rut.value, self.selector.strategy_stats)
        for strategy in test_strategies:
            self.assertIn(strategy.value, self.selector.strategy_stats[test_rut.value])
    
    def test_update_stats(self):
        """Test updating stats for a strategy."""
        # Update with a success
        self.selector.update_stats(self.rut_type, self.strategies[0], True)
        
        stats = self.selector.strategy_stats[self.rut_type.value][self.strategies[0].value]
        self.assertEqual(stats["plays"], 1)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["average_reward"], 0.5 * 0.8 + 1.0 * 0.2)  # Based on learning rate
        
        # Update with a failure
        self.selector.update_stats(self.rut_type, self.strategies[1], False)
        
        stats = self.selector.strategy_stats[self.rut_type.value][self.strategies[1].value]
        self.assertEqual(stats["plays"], 1)
        self.assertEqual(stats["wins"], 0)
        self.assertEqual(stats["average_reward"], 0.5 * 0.8 + 0.0 * 0.2)  # Based on learning rate
        
        # Update with a custom reward
        self.selector.update_stats(self.rut_type, self.strategies[2], True, 0.75)
        
        stats = self.selector.strategy_stats[self.rut_type.value][self.strategies[2].value]
        self.assertEqual(stats["plays"], 1)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["average_reward"], 0.5 * 0.8 + 0.75 * 0.2)  # Based on learning rate
    
    def test_epsilon_greedy_selection(self):
        """Test epsilon-greedy strategy selection."""
        # Force exploration path
        with patch('random.random', return_value=0.1):  # Below exploration factor
            self.selector.algorithm = "epsilon_greedy"
            
            # Mock random.choice to return predictable result
            with patch('random.choice', return_value=self.strategies[0].value):
                selected = self.selector.select_strategy(self.rut_type, self.strategies)
                self.assertEqual(selected, self.strategies[0])
        
        # Force exploitation path
        with patch('random.random', return_value=0.9):  # Above exploration factor
            self.selector.algorithm = "epsilon_greedy"
            
            # Set rewards to clearly identify best strategy
            self.selector.strategy_stats[self.rut_type.value][self.strategies[0].value]["average_reward"] = 0.9
            self.selector.strategy_stats[self.rut_type.value][self.strategies[1].value]["average_reward"] = 0.5
            self.selector.strategy_stats[self.rut_type.value][self.strategies[2].value]["average_reward"] = 0.7
            
            selected = self.selector.select_strategy(self.rut_type, self.strategies)
            self.assertEqual(selected, self.strategies[0])
    
    def test_softmax_selection(self):
        """Test softmax strategy selection."""
        self.selector.algorithm = "softmax"
        
        # Set rewards to clearly differentiate
        self.selector.strategy_stats[self.rut_type.value][self.strategies[0].value]["average_reward"] = 0.9
        self.selector.strategy_stats[self.rut_type.value][self.strategies[1].value]["average_reward"] = 0.1
        self.selector.strategy_stats[self.rut_type.value][self.strategies[2].value]["average_reward"] = 0.5
        
        # Mock random.choices to ensure deterministic selection
        # For softmax with these values, probabilities should heavily favor the first strategy
        with patch('random.choices', return_value=[0]):  # Select first strategy
            selected = self.selector.select_strategy(self.rut_type, self.strategies)
            self.assertEqual(selected, self.strategies[0])
    
    def test_ucb_selection_no_plays(self):
        """Test UCB selection with no plays."""
        self.selector.algorithm = "ucb"
        
        # Reset all plays to 0
        for strat in self.strategies:
            self.selector.strategy_stats[self.rut_type.value][strat.value]["plays"] = 0
        
        # Mock random.choice for predictable result
        with patch('random.choice', return_value=self.strategies[0]):
            selected = self.selector.select_strategy(self.rut_type, self.strategies)
            self.assertEqual(selected, self.strategies[0])
    
    def test_ucb_selection_with_plays(self):
        """Test UCB selection with existing plays."""
        self.selector.algorithm = "ucb"
        
        # Set up strategy stats to test UCB formula
        self.selector.strategy_stats[self.rut_type.value][self.strategies[0].value] = {
            "plays": 10, 
            "wins": 7, 
            "average_reward": 0.7
        }
        self.selector.strategy_stats[self.rut_type.value][self.strategies[1].value] = {
            "plays": 5, 
            "wins": 2, 
            "average_reward": 0.4
        }
        self.selector.strategy_stats[self.rut_type.value][self.strategies[2].value] = {
            "plays": 2, 
            "wins": 1, 
            "average_reward": 0.5
        }
        
        selected = self.selector.select_strategy(self.rut_type, self.strategies)
        
        # Manually calculate UCB scores
        total_plays = 17
        
        # UCB for strategy 0: exploit + explore = 0.7 + 0.3 * sqrt(2 * ln(17) / 10)
        ucb0 = 0.7 + 0.3 * math.sqrt(2 * math.log(total_plays) / 10)
        
        # UCB for strategy 1: exploit + explore = 0.4 + 0.3 * sqrt(2 * ln(17) / 5)
        ucb1 = 0.4 + 0.3 * math.sqrt(2 * math.log(total_plays) / 5)
        
        # UCB for strategy 2: exploit + explore = 0.5 + 0.3 * sqrt(2 * ln(17) / 2)
        ucb2 = 0.5 + 0.3 * math.sqrt(2 * math.log(total_plays) / 2)
        
        # Determine which strategy has highest UCB
        ucb_scores = [ucb0, ucb1, ucb2]
        expected_index = ucb_scores.index(max(ucb_scores))
        
        self.assertEqual(selected, self.strategies[expected_index])
    
    def test_multiple_updates(self):
        """Test the effect of multiple updates on strategy selection."""
        # Start with UCB algorithm
        self.selector.algorithm = "ucb"
        
        # Simulate successful uses of strategy 0
        for _ in range(5):
            self.selector.update_stats(self.rut_type, self.strategies[0], True)
        
        # Simulate mixed results for strategy 1
        for i in range(10):
            self.selector.update_stats(self.rut_type, self.strategies[1], i % 2 == 0)
        
        # Simulate poor results for strategy 2
        for _ in range(7):
            self.selector.update_stats(self.rut_type, self.strategies[2], False)
        
        # Check rewards after updates
        stats0 = self.selector.strategy_stats[self.rut_type.value][self.strategies[0].value]
        stats1 = self.selector.strategy_stats[self.rut_type.value][self.strategies[1].value]
        stats2 = self.selector.strategy_stats[self.rut_type.value][self.strategies[2].value]
        
        # Verify plays and wins
        self.assertEqual(stats0["plays"], 5)
        self.assertEqual(stats0["wins"], 5)
        
        self.assertEqual(stats1["plays"], 10)
        self.assertEqual(stats1["wins"], 5)
        
        self.assertEqual(stats2["plays"], 7)
        self.assertEqual(stats2["wins"], 0)
        
        # Select strategy and verify it's the best performer
        selected = self.selector.select_strategy(self.rut_type, self.strategies)
        
        # Calculate expected UCB scores
        total_plays = 22
        
        # With more plays, the exploration bonus diminishes, so best average reward
        # should be the primary factor in selection
        best_strategy_index = max(
            range(len(self.strategies)), 
            key=lambda i: self.selector.strategy_stats[self.rut_type.value][self.strategies[i].value]["average_reward"]
        )
        
        self.assertEqual(selected, self.strategies[best_strategy_index])


if __name__ == "__main__":
    unittest.main() 