"""
Pattern optimization system for enhancing learning from recurring interactions.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

class PatternType(Enum):
    TEMPORAL = "temporal"  # Time-based patterns
    BEHAVIORAL = "behavioral"  # Interaction patterns
    COGNITIVE = "cognitive"  # Learning patterns
    ADAPTIVE = "adaptive"  # Response patterns
    META = "meta"  # Pattern of patterns

@dataclass
class PatternContext:
    """Context for pattern analysis."""
    frequency: float
    consistency: float
    evolution_rate: float
    adaptability: float
    effectiveness: float

@dataclass
class OptimizationMetrics:
    """Metrics for pattern optimization."""
    learning_rate: float
    adaptation_speed: float
    pattern_stability: float
    effectiveness_gain: float
    meta_learning: float

class PatternOptimizer:
    def __init__(self):
        """Initialize pattern optimization system."""
        self.pattern_history: Dict[str, List[Dict]] = {}
        self.optimization_metrics: Dict[str, Dict[PatternType, float]] = {}
        self.learning_insights: Dict[str, List[Dict]] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent optimizer state."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_optimizer"))
        data_file = data_dir / "optimizer_state.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    self.pattern_history = data.get('pattern_history', {})

                    # Restore optimization metrics
                    self.optimization_metrics = {
                        domain: {
                            PatternType(p): v
                            for p, v in patterns.items()
                        }
                        for domain, patterns in data.get(
                            'optimization_metrics', {}).items()
                    }

                    self.learning_insights = data.get('learning_insights', {})
            except Exception as e:
                logger.error(f"Error loading optimizer state: {e}")

    def _save_state(self) -> None:
        """Save optimizer state persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_optimizer"))
        data_file = data_dir / "optimizer_state.json"

        try:
            data = {
                'pattern_history': self.pattern_history,
                'optimization_metrics': {
                    domain: {
                        p.value: v
                        for p, v in patterns.items()
                    }
                    for domain, patterns in self.optimization_metrics.items()
                },
                'learning_insights': self.learning_insights
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving optimizer state: {e}")

    def analyze_pattern(self, pattern_id: str,
                       context: PatternContext) -> Dict:
        """
        Analyze pattern and generate optimization insights.
        
        Args:
            pattern_id: Pattern identifier
            context: Pattern context
            
        Returns:
            Dict containing analysis results
        """
        # Analyze pattern characteristics
        pattern_type = self._determine_pattern_type(context)

        # Score optimization potential
        optimization_scores = self._score_optimization_potential(
            pattern_type, context)

        # Generate optimization strategy
        optimization_strategy = self._generate_optimization_strategy(
            optimization_scores, context)

        # Record analysis
        if pattern_id not in self.pattern_history:
            self.pattern_history[pattern_id] = []

        self.pattern_history[pattern_id].append({
            'timestamp': datetime.now().isoformat(),
            'context': asdict(context),
            'pattern_type': pattern_type.value,
            'optimization_scores': {
                k.value: v for k, v in optimization_scores.items()
            }
        })

        self._save_state()

        return {
            'pattern_type': pattern_type.value,
            'optimization_scores': {
                k.value: v for k, v in optimization_scores.items()
            },
            'optimization_strategy': optimization_strategy
        }

    def _determine_pattern_type(self,
                              context: PatternContext) -> PatternType:
        """Determine pattern type from context."""
        # Calculate type scores
        type_scores = {
            PatternType.TEMPORAL: (
                context.frequency * 0.4 +
                context.consistency * 0.6
            ),
            PatternType.BEHAVIORAL: (
                context.consistency * 0.3 +
                context.effectiveness * 0.7
            ),
            PatternType.COGNITIVE: (
                context.evolution_rate * 0.6 +
                context.effectiveness * 0.4
            ),
            PatternType.ADAPTIVE: (
                context.adaptability * 0.7 +
                context.evolution_rate * 0.3
            ),
            PatternType.META: (
                context.evolution_rate * 0.4 +
                context.meta_learning * 0.6
            )
        }

        # Return highest scoring type
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _score_optimization_potential(self,
                                   pattern_type: PatternType,
                                   context: PatternContext
                                   ) -> Dict[PatternType, float]:
        """Score optimization potential for different approaches."""
        base_scores = {
            PatternType.TEMPORAL: 0.5,
            PatternType.BEHAVIORAL: 0.5,
            PatternType.COGNITIVE: 0.5,
            PatternType.ADAPTIVE: 0.5,
            PatternType.META: 0.5
        }

        # Adjust for pattern type affinity
        base_scores[pattern_type] += 0.2

        # Adjust for context factors
        if context.frequency > 0.7:
            base_scores[PatternType.TEMPORAL] += 0.2
            base_scores[PatternType.BEHAVIORAL] += 0.1

        if context.evolution_rate > 0.7:
            base_scores[PatternType.COGNITIVE] += 0.2
            base_scores[PatternType.META] += 0.2

        if context.adaptability > 0.7:
            base_scores[PatternType.ADAPTIVE] += 0.2
            base_scores[PatternType.META] += 0.1

        # Normalize scores
        total = sum(base_scores.values())
        return {
            pattern_type: score / total
            for pattern_type, score in base_scores.items()
        }

    def _generate_optimization_strategy(self,
                                     optimization_scores: Dict[PatternType, float],
                                     context: PatternContext) -> List[Dict]:
        """Generate optimization strategy based on scores."""
        strategy = []

        # Sort optimization approaches by score
        sorted_approaches = sorted(
            optimization_scores.items(), key=lambda x: x[1], reverse=True)

        for approach, score in sorted_approaches:
            if score < 0.1:  # Skip low-scoring approaches
                continue

            if approach == PatternType.TEMPORAL:
                strategy.append({
                    'approach': 'temporal_optimization',
                    'weight': score,
                    'actions': [
                        'Analyze timing patterns',
                        'Optimize frequency',
                        'Enhance consistency'
                    ]
                })

            elif approach == PatternType.BEHAVIORAL:
                strategy.append({
                    'approach': 'behavioral_optimization',
                    'weight': score,
                    'actions': [
                        'Analyze interaction patterns',
                        'Optimize responses',
                        'Enhance effectiveness'
                    ]
                })

            elif approach == PatternType.COGNITIVE:
                strategy.append({
                    'approach': 'cognitive_optimization',
                    'weight': score,
                    'actions': [
                        'Analyze learning patterns',
                        'Optimize understanding',
                        'Enhance knowledge integration'
                    ]
                })

            elif approach == PatternType.ADAPTIVE:
                strategy.append({
                    'approach': 'adaptive_optimization',
                    'weight': score,
                    'actions': [
                        'Analyze adaptation patterns',
                        'Optimize flexibility',
                        'Enhance responsiveness'
                    ]
                })

            elif approach == PatternType.META:
                strategy.append({
                    'approach': 'meta_optimization',
                    'weight': score,
                    'actions': [
                        'Analyze pattern relationships',
                        'Optimize pattern learning',
                        'Enhance meta-understanding'
                    ]
                })

        return strategy

    def update_optimization_results(self, pattern_id: str,
                                  metrics: OptimizationMetrics) -> Dict:
        """
        Update optimization results and learning insights.
        
        Args:
            pattern_id: Pattern identifier
            metrics: Optimization metrics
            
        Returns:
            Dict containing update results
        """
        if pattern_id not in self.pattern_history:
            return {
                'status': 'error',
                'message': 'No pattern history found'
            }

        # Get latest pattern analysis
        latest_pattern = self.pattern_history[pattern_id][-1]
        pattern_type = PatternType(latest_pattern['pattern_type'])

        # Update optimization metrics
        if pattern_id not in self.optimization_metrics:
            self.optimization_metrics[pattern_id] = {
                pattern_type: 0.5 for pattern_type in PatternType
            }

        # Update with new metrics
        current_metrics = self.optimization_metrics[pattern_id]
        learning_impact = (
            metrics.learning_rate * 0.3 +
            metrics.adaptation_speed * 0.2 +
            metrics.pattern_stability * 0.2 +
            metrics.effectiveness_gain * 0.2 +
            metrics.meta_learning * 0.1
        )

        for p_type in PatternType:
            current_value = current_metrics[p_type]
            if p_type == pattern_type:
                # Higher weight for matching type
                new_value = current_value * 0.7 + learning_impact * 0.3
            else:
                # Lower weight for other types
                new_value = current_value * 0.9 + learning_impact * 0.1
            current_metrics[p_type] = new_value

        # Update learning insights
        if pattern_id not in self.learning_insights:
            self.learning_insights[pattern_id] = []

        self.learning_insights[pattern_id].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'pattern_type': pattern_type.value,
            'learning_impact': learning_impact
        })

        # Keep last 100 insights
        if len(self.learning_insights[pattern_id]) > 100:
            self.learning_insights[pattern_id] = (
                self.learning_insights[pattern_id][-100:]
            )

        self._save_state()

        return {
            'status': 'updated',
            'pattern_type': pattern_type.value,
            'learning_impact': learning_impact,
            'optimization_progress': len(self.learning_insights[pattern_id]) / 100
        }

    def get_optimization_insights(self) -> Dict:
        """
        Get insights about pattern optimization effectiveness.
        
        Returns:
            Dict containing optimization insights
        """
        # Analyze pattern effectiveness
        pattern_effectiveness = {}
        for pattern_id, metrics in self.optimization_metrics.items():
            avg_effectiveness = sum(metrics.values()) / len(metrics)
            most_effective = max(metrics.items(), key=lambda x: x[1])
            pattern_effectiveness[pattern_id] = {
                'average_effectiveness': avg_effectiveness,
                'best_approach': most_effective[0].value,
                'best_score': most_effective[1]
            }

        # Analyze learning progress
        learning_progress = {}
        for pattern_id, insights in self.learning_insights.items():
            if insights:
                recent_insights = insights[-10:]
                avg_impact = sum(
                    i['learning_impact'] for i in recent_insights
                ) / len(recent_insights)
                learning_progress[pattern_id] = {
                    'insights_recorded': len(insights),
                    'recent_impact': avg_impact
                }

        # Calculate meta-learning metrics
        meta_learning = {
            'temporal_stability': self._calculate_stability(PatternType.TEMPORAL),
            'behavioral_adaptation': self._calculate_stability(
                PatternType.BEHAVIORAL),
            'cognitive_growth': self._calculate_stability(PatternType.COGNITIVE),
            'adaptive_flexibility': self._calculate_stability(PatternType.ADAPTIVE),
            'meta_understanding': self._calculate_stability(PatternType.META)
        }

        return {
            'total_patterns': len(self.pattern_history),
            'pattern_effectiveness': pattern_effectiveness,
            'learning_progress': learning_progress,
            'meta_learning': meta_learning
        }

    def _calculate_stability(self, pattern_type: PatternType) -> float:
        """Calculate stability metric for pattern type."""
        if not self.optimization_metrics:
            return 0.0

        # Get effectiveness values for pattern type
        type_effectiveness = [
            metrics[pattern_type]
            for metrics in self.optimization_metrics.values()
        ]

        if not type_effectiveness:
            return 0.0

        # Calculate stability as consistency of effectiveness
        mean_effectiveness = sum(type_effectiveness) / len(type_effectiveness)
        variations = [
            abs(v - mean_effectiveness) for v in type_effectiveness
        ]

        if not variations:
            return 1.0

        stability = 1.0 - (sum(variations) / len(variations))
        return max(0.0, min(1.0, stability))
