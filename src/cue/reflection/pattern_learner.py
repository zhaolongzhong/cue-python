"""
Self-reflection and meta-learning system for analyzing interaction patterns.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class PatternLearner:
    def __init__(self):
        """Initialize pattern learning system."""
        self.learning_history: Dict[str, List[Dict]] = {}
        self.adaptation_metrics: Dict[str, Dict] = {}
        self.reflection_insights: List[Dict] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent learning state."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_learner"))
        data_file = data_dir / "learning_state.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    self.learning_history = data.get('learning_history', {})
                    self.adaptation_metrics = data.get('adaptation_metrics', {})
                    self.reflection_insights = data.get('reflection_insights', [])
            except Exception as e:
                logger.error(f"Error loading learning state: {e}")

    def _save_state(self) -> None:
        """Save learning state persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_learner"))
        data_file = data_dir / "learning_state.json"

        try:
            data = {
                'learning_history': self.learning_history,
                'adaptation_metrics': self.adaptation_metrics,
                'reflection_insights': self.reflection_insights
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def analyze_pattern(self, pattern_id: str,
                       pattern_data: Dict,
                       effectiveness: float) -> Dict:
        """
        Analyze pattern and generate learning insights.
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_data: Pattern analysis data
            effectiveness: Pattern recognition effectiveness
            
        Returns:
            Dict containing analysis results and insights
        """
        # Record learning event
        learning_event = {
            'timestamp': datetime.now().isoformat(),
            'effectiveness': effectiveness,
            'pattern_data': pattern_data
        }

        if pattern_id not in self.learning_history:
            self.learning_history[pattern_id] = []
        self.learning_history[pattern_id].append(learning_event)

        # Analyze learning progress
        learning_curve = self._analyze_learning_curve(pattern_id)
        adaptation_rate = self._calculate_adaptation_rate(pattern_id)

        # Update adaptation metrics
        self.adaptation_metrics[pattern_id] = {
            'learning_rate': learning_curve['learning_rate'],
            'stability': learning_curve['stability'],
            'adaptation_rate': adaptation_rate,
            'last_updated': datetime.now().isoformat()
        }

        # Generate reflection insight
        insight = self._generate_insight(pattern_id, learning_curve, adaptation_rate)
        if insight:
            self.reflection_insights.append(insight)

        self._save_state()

        return {
            'learning_curve': learning_curve,
            'adaptation_rate': adaptation_rate,
            'latest_insight': insight
        }

    def _analyze_learning_curve(self, pattern_id: str) -> Dict:
        """Analyze learning progress over time."""
        if pattern_id not in self.learning_history:
            return {'learning_rate': 0.0, 'stability': 0.0}

        history = self.learning_history[pattern_id]
        if len(history) < 2:
            return {'learning_rate': 0.0, 'stability': 0.0}

        # Calculate learning rate from effectiveness trend
        effectiveness_values = [
            event['effectiveness'] for event in history[-10:]  # Last 10 events
        ]

        if len(effectiveness_values) < 2:
            return {'learning_rate': 0.0, 'stability': 0.0}

        # Calculate learning rate
        learning_rate = (effectiveness_values[-1] - effectiveness_values[0]) / len(effectiveness_values)

        # Calculate stability
        variations = [
            abs(effectiveness_values[i] - effectiveness_values[i-1])
            for i in range(1, len(effectiveness_values))
        ]
        stability = 1.0 - (sum(variations) / len(variations))

        return {
            'learning_rate': learning_rate,
            'stability': stability
        }

    def _calculate_adaptation_rate(self, pattern_id: str) -> float:
        """Calculate how quickly the system adapts to pattern changes."""
        if pattern_id not in self.learning_history:
            return 0.0

        history = self.learning_history[pattern_id]
        if len(history) < 3:
            return 0.0

        # Look at effectiveness recovery after drops
        effectiveness_values = [
            event['effectiveness'] for event in history
        ]

        recovery_rates = []
        for i in range(2, len(effectiveness_values)):
            prev_value = effectiveness_values[i-1]
            curr_value = effectiveness_values[i]

            if prev_value < effectiveness_values[i-2]:  # Detected drop
                if curr_value > prev_value:  # Started recovery
                    recovery_rate = (curr_value - prev_value) / prev_value
                    recovery_rates.append(recovery_rate)

        if not recovery_rates:
            return 0.0

        return sum(recovery_rates) / len(recovery_rates)

    def _generate_insight(self, pattern_id: str,
                         learning_curve: Dict,
                         adaptation_rate: float) -> Optional[Dict]:
        """Generate reflection insight from analysis."""
        now = datetime.now()

        # Get historical metrics
        prev_metrics = self.adaptation_metrics.get(pattern_id, {})
        prev_learning_rate = prev_metrics.get('learning_rate', 0.0)
        prev_stability = prev_metrics.get('stability', 0.0)

        # Calculate changes
        learning_rate_change = learning_curve['learning_rate'] - prev_learning_rate
        stability_change = learning_curve['stability'] - prev_stability

        # Generate insight if significant changes detected
        if (abs(learning_rate_change) > 0.1 or
            abs(stability_change) > 0.1 or
            adaptation_rate > 0.2):

            insight_type = self._determine_insight_type(
                learning_rate_change, stability_change, adaptation_rate)

            return {
                'timestamp': now.isoformat(),
                'pattern_id': pattern_id,
                'type': insight_type,
                'metrics': {
                    'learning_rate_change': learning_rate_change,
                    'stability_change': stability_change,
                    'adaptation_rate': adaptation_rate
                },
                'description': self._format_insight_description(
                    insight_type, learning_rate_change,
                    stability_change, adaptation_rate
                )
            }

        return None

    def _determine_insight_type(self, learning_rate_change: float,
                              stability_change: float,
                              adaptation_rate: float) -> str:
        """Determine the type of learning insight."""
        if learning_rate_change > 0.1 and stability_change > 0:
            return 'accelerated_learning'
        elif adaptation_rate > 0.2:
            return 'rapid_adaptation'
        elif stability_change > 0.1:
            return 'increased_stability'
        elif learning_rate_change < -0.1:
            return 'learning_plateau'
        elif stability_change < -0.1:
            return 'stability_decline'
        else:
            return 'steady_progress'

    def _format_insight_description(self, insight_type: str,
                                  learning_rate_change: float,
                                  stability_change: float,
                                  adaptation_rate: float) -> str:
        """Format human-readable insight description."""
        descriptions = {
            'accelerated_learning': (
                f"Learning rate increased by {learning_rate_change:.2f} with "
                f"improved stability of {stability_change:.2f}"
            ),
            'rapid_adaptation': (
                f"Demonstrated quick adaptation with rate of {adaptation_rate:.2f}"
            ),
            'increased_stability': (
                f"Pattern recognition stability improved by {stability_change:.2f}"
            ),
            'learning_plateau': (
                f"Learning rate decreased by {abs(learning_rate_change):.2f}, "
                "indicating potential plateau"
            ),
            'stability_decline': (
                f"Pattern stability declined by {abs(stability_change):.2f}, "
                "requiring attention"
            ),
            'steady_progress': (
                f"Maintaining steady progress with adaptation rate "
                f"of {adaptation_rate:.2f}"
            )
        }

        return descriptions.get(insight_type, "Insight details unavailable")

    def get_learning_summary(self, pattern_id: str) -> Dict:
        """
        Get summary of learning progress for a pattern.
        
        Args:
            pattern_id: Pattern to summarize
            
        Returns:
            Dict containing learning summary
        """
        if pattern_id not in self.learning_history:
            return {
                'status': 'no_data',
                'message': 'No learning history available'
            }

        history = self.learning_history[pattern_id]
        metrics = self.adaptation_metrics.get(pattern_id, {})

        recent_insights = [
            insight for insight in self.reflection_insights[-5:]
            if insight['pattern_id'] == pattern_id
        ]

        return {
            'status': 'active',
            'events_recorded': len(history),
            'latest_effectiveness': history[-1]['effectiveness'],
            'learning_rate': metrics.get('learning_rate', 0.0),
            'stability': metrics.get('stability', 0.0),
            'adaptation_rate': metrics.get('adaptation_rate', 0.0),
            'recent_insights': recent_insights
        }
