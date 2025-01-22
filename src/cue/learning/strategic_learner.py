"""
Strategic learning coordinator for optimizing self-improvement and pattern adaptation.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    EXPLORATORY = "exploratory"  # Try new approaches
    EXPLOITATIVE = "exploitative"  # Optimize known patterns
    BALANCED = "balanced"  # Mix of exploration and exploitation
    RECOVERY = "recovery"  # Focus on stability after decline
    ACCELERATED = "accelerated"  # Rapid learning mode

@dataclass
class LearningState:
    """Represents the current learning state."""
    strategy: LearningStrategy
    effectiveness: float
    stability: float
    adaptability: float
    exploration_rate: float
    last_updated: str

@dataclass
class StrategyMetrics:
    """Metrics for evaluating strategy effectiveness."""
    success_rate: float
    recovery_speed: float
    exploration_value: float
    stability_impact: float

class StrategicLearner:
    def __init__(self):
        """Initialize strategic learning system."""
        self.learning_states: Dict[str, LearningState] = {}
        self.strategy_history: Dict[str, List[Dict]] = {}
        self.strategy_metrics: Dict[str, Dict[LearningStrategy, StrategyMetrics]] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent learning state."""
        data_dir = Path(os.path.expanduser("~/.cache/strategic_learner"))
        data_file = data_dir / "learning_state.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)

                    # Restore learning states
                    self.learning_states = {
                        k: LearningState(
                            strategy=LearningStrategy(v['strategy']),
                            effectiveness=v['effectiveness'],
                            stability=v['stability'],
                            adaptability=v['adaptability'],
                            exploration_rate=v['exploration_rate'],
                            last_updated=v['last_updated']
                        )
                        for k, v in data.get('learning_states', {}).items()
                    }

                    self.strategy_history = data.get('strategy_history', {})

                    # Restore strategy metrics
                    self.strategy_metrics = {
                        k: {
                            LearningStrategy(sk): StrategyMetrics(**sv)
                            for sk, sv in v.items()
                        }
                        for k, v in data.get('strategy_metrics', {}).items()
                    }
            except Exception as e:
                logger.error(f"Error loading learning state: {e}")

    def _save_state(self) -> None:
        """Save learning state persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/strategic_learner"))
        data_file = data_dir / "learning_state.json"

        try:
            data = {
                'learning_states': {
                    k: asdict(v) for k, v in self.learning_states.items()
                },
                'strategy_history': self.strategy_history,
                'strategy_metrics': {
                    k: {
                        sk.value: asdict(sv)
                        for sk, sv in v.items()
                    }
                    for k, v in self.strategy_metrics.items()
                }
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def update_learning_state(self, domain_id: str,
                            metrics: Dict[str, float]) -> Dict:
        """
        Update learning state and adapt strategy.
        
        Args:
            domain_id: Learning domain identifier
            metrics: Current performance metrics
            
        Returns:
            Dict containing updated state and recommendations
        """
        current_state = self.learning_states.get(domain_id)

        # Calculate new state
        new_state = LearningState(
            strategy=self._determine_strategy(domain_id, metrics, current_state),
            effectiveness=metrics.get('effectiveness', 0.0),
            stability=metrics.get('stability', 0.0),
            adaptability=metrics.get('adaptability', 0.0),
            exploration_rate=self._calculate_exploration_rate(
                domain_id, metrics, current_state),
            last_updated=datetime.now().isoformat()
        )

        # Record state change
        self.learning_states[domain_id] = new_state

        # Update history
        if domain_id not in self.strategy_history:
            self.strategy_history[domain_id] = []

        self.strategy_history[domain_id].append({
            'timestamp': new_state.last_updated,
            'strategy': new_state.strategy.value,
            'metrics': metrics
        })

        # Generate recommendations
        recommendations = self._generate_recommendations(domain_id, new_state)

        self._save_state()

        return {
            'state': asdict(new_state),
            'recommendations': recommendations
        }

    def _determine_strategy(self, domain_id: str,
                          metrics: Dict[str, float],
                          current_state: Optional[LearningState]) -> LearningStrategy:
        """Determine optimal learning strategy based on current state."""
        if not current_state:
            return LearningStrategy.EXPLORATORY

        effectiveness = metrics.get('effectiveness', 0.0)
        stability = metrics.get('stability', 0.0)
        adaptability = metrics.get('adaptability', 0.0)

        # Get strategy effectiveness metrics
        strategy_performance = self.strategy_metrics.get(domain_id, {})

        # Check for significant decline
        if effectiveness < current_state.effectiveness * 0.8:
            return LearningStrategy.RECOVERY

        # Check for rapid improvement opportunity
        if (effectiveness > 0.8 and stability > 0.7 and
            current_state.strategy != LearningStrategy.ACCELERATED):
            return LearningStrategy.ACCELERATED

        # Evaluate exploration vs exploitation
        if self._should_explore(domain_id, metrics, current_state):
            return LearningStrategy.EXPLORATORY

        # Default to balanced approach
        return LearningStrategy.BALANCED

    def _calculate_exploration_rate(self, domain_id: str,
                                  metrics: Dict[str, float],
                                  current_state: Optional[LearningState]) -> float:
        """Calculate optimal exploration rate."""
        if not current_state:
            return 0.8  # High initial exploration

        base_rate = current_state.exploration_rate

        # Adjust based on effectiveness trend
        effectiveness_delta = metrics.get('effectiveness', 0.0) - current_state.effectiveness

        if effectiveness_delta > 0.1:
            # Reduce exploration when improving
            base_rate *= 0.9
        elif effectiveness_delta < -0.1:
            # Increase exploration when declining
            base_rate = min(1.0, base_rate * 1.2)

        # Adjust based on stability
        stability = metrics.get('stability', 0.0)
        if stability < 0.5:
            # Reduce exploration when unstable
            base_rate *= 0.8

        return min(1.0, max(0.1, base_rate))

    def _should_explore(self, domain_id: str,
                       metrics: Dict[str, float],
                       current_state: LearningState) -> bool:
        """Determine if exploration is beneficial."""
        if domain_id not in self.strategy_metrics:
            return True

        current_metrics = self.strategy_metrics[domain_id]

        # Calculate exploration value
        exploration_value = current_metrics.get(
            LearningStrategy.EXPLORATORY,
            StrategyMetrics(0.5, 0.5, 0.5, 0.5)
        ).exploration_value

        # Check if exploration has been valuable
        if exploration_value > 0.7:
            return True

        # Check if current strategy is stagnating
        if (current_state.strategy != LearningStrategy.EXPLORATORY and
            metrics.get('effectiveness', 0.0) < 0.6):
            return True

        return False

    def _generate_recommendations(self, domain_id: str,
                                state: LearningState) -> List[Dict]:
        """Generate strategic recommendations."""
        recommendations = []

        # Strategy-specific recommendations
        strategy_recs = {
            LearningStrategy.EXPLORATORY: [
                {
                    'type': 'exploration',
                    'action': 'Increase pattern variation',
                    'confidence': 0.8
                },
                {
                    'type': 'monitoring',
                    'action': 'Track novel pattern effectiveness',
                    'confidence': 0.9
                }
            ],
            LearningStrategy.EXPLOITATIVE: [
                {
                    'type': 'optimization',
                    'action': 'Fine-tune successful patterns',
                    'confidence': 0.85
                },
                {
                    'type': 'analysis',
                    'action': 'Detailed performance analysis',
                    'confidence': 0.9
                }
            ],
            LearningStrategy.BALANCED: [
                {
                    'type': 'balance',
                    'action': 'Maintain exploration/exploitation ratio',
                    'confidence': 0.75
                }
            ],
            LearningStrategy.RECOVERY: [
                {
                    'type': 'stability',
                    'action': 'Focus on pattern stability',
                    'confidence': 0.9
                },
                {
                    'type': 'analysis',
                    'action': 'Analyze performance decline',
                    'confidence': 0.85
                }
            ],
            LearningStrategy.ACCELERATED: [
                {
                    'type': 'optimization',
                    'action': 'Accelerate successful patterns',
                    'confidence': 0.9
                },
                {
                    'type': 'monitoring',
                    'action': 'Close performance monitoring',
                    'confidence': 0.85
                }
            ]
        }

        # Add strategy-specific recommendations
        recommendations.extend(strategy_recs.get(state.strategy, []))

        # Add stability-based recommendations
        if state.stability < 0.5:
            recommendations.append({
                'type': 'stability',
                'action': 'Improve pattern stability',
                'confidence': 0.8
            })

        # Add exploration-based recommendations
        if state.exploration_rate > 0.7:
            recommendations.append({
                'type': 'exploration',
                'action': 'Document exploration outcomes',
                'confidence': 0.75
            })

        return recommendations

    def update_strategy_metrics(self, domain_id: str,
                              strategy: LearningStrategy,
                              metrics: Dict[str, float]) -> None:
        """Update effectiveness metrics for a strategy."""
        if domain_id not in self.strategy_metrics:
            self.strategy_metrics[domain_id] = {}

        current_metrics = self.strategy_metrics[domain_id].get(
            strategy,
            StrategyMetrics(0.5, 0.5, 0.5, 0.5)
        )

        # Calculate new metrics
        success_rate = (
            current_metrics.success_rate * 0.7 +
            metrics.get('effectiveness', 0.0) * 0.3
        )

        recovery_speed = (
            current_metrics.recovery_speed * 0.7 +
            metrics.get('adaptability', 0.0) * 0.3
        )

        exploration_value = (
            current_metrics.exploration_value * 0.7 +
            (metrics.get('effectiveness', 0.0) *
             metrics.get('stability', 0.0)) * 0.3
        )

        stability_impact = (
            current_metrics.stability_impact * 0.7 +
            metrics.get('stability', 0.0) * 0.3
        )

        # Update metrics
        self.strategy_metrics[domain_id][strategy] = StrategyMetrics(
            success_rate=success_rate,
            recovery_speed=recovery_speed,
            exploration_value=exploration_value,
            stability_impact=stability_impact
        )

        self._save_state()

    def get_learning_summary(self, domain_id: str) -> Dict:
        """
        Get summary of learning progress and strategy effectiveness.
        
        Args:
            domain_id: Learning domain to summarize
            
        Returns:
            Dict containing learning summary
        """
        if domain_id not in self.learning_states:
            return {
                'status': 'no_data',
                'message': 'No learning data available'
            }

        current_state = self.learning_states[domain_id]
        history = self.strategy_history.get(domain_id, [])
        metrics = self.strategy_metrics.get(domain_id, {})

        # Calculate strategy effectiveness
        strategy_effectiveness = {
            strategy.value: asdict(metrics.get(strategy, StrategyMetrics(0, 0, 0, 0)))
            for strategy in LearningStrategy
        }

        # Get recent strategies
        recent_strategies = [
            entry['strategy'] for entry in history[-5:]
        ] if history else []

        return {
            'status': 'active',
            'current_state': asdict(current_state),
            'strategy_effectiveness': strategy_effectiveness,
            'recent_strategies': recent_strategies,
            'total_transitions': len(history),
            'metrics_summary': {
                strategy.value: {
                    'success_rate': metrics.get(
                        strategy, StrategyMetrics(0, 0, 0, 0)
                    ).success_rate
                }
                for strategy in LearningStrategy
            }
        }
