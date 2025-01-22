"""
Pattern-aware coordination system for multi-agent interactions.
Enhances agent responses based on detected interaction patterns.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PatternCoordinator:
    def __init__(self):
        """Initialize the pattern-aware coordinator."""
        self.interaction_patterns: Dict[str, List[Tuple[datetime, str, float]]] = {}
        self.agent_patterns: Dict[str, Dict[str, float]] = {}
        self.pattern_scores: Dict[str, float] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load persistent pattern data."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_coordinator"))
        data_file = data_dir / "patterns.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    self.pattern_scores = data.get('pattern_scores', {})

                    # Restore interaction patterns with datetime
                    self.interaction_patterns = {
                        k: [(datetime.fromisoformat(d), m, s) for d, m, s in v]
                        for k, v in data.get('interaction_patterns', {}).items()
                    }

                    self.agent_patterns = data.get('agent_patterns', {})
            except Exception as e:
                logger.error(f"Error loading pattern data: {e}")

    def _save_patterns(self) -> None:
        """Save pattern data persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/pattern_coordinator"))
        data_file = data_dir / "patterns.json"

        try:
            data = {
                'pattern_scores': self.pattern_scores,
                'interaction_patterns': {
                    k: [(d.isoformat(), m, s) for d, m, s in v]
                    for k, v in self.interaction_patterns.items()
                },
                'agent_patterns': self.agent_patterns
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving pattern data: {e}")

    def record_interaction(self, pattern_id: str, message: str,
                         agent_id: Optional[str] = None) -> None:
        """
        Record an interaction pattern occurrence.
        
        Args:
            pattern_id: Unique identifier for the pattern
            message: The interaction message
            agent_id: Optional agent identifier for multi-agent patterns
        """
        now = datetime.now()

        # Record interaction pattern
        if pattern_id not in self.interaction_patterns:
            self.interaction_patterns[pattern_id] = []
        self.interaction_patterns[pattern_id].append((now, message, 1.0))

        # Update agent patterns if applicable
        if agent_id:
            if agent_id not in self.agent_patterns:
                self.agent_patterns[agent_id] = {}
            if pattern_id not in self.agent_patterns[agent_id]:
                self.agent_patterns[agent_id][pattern_id] = 0
            self.agent_patterns[agent_id][pattern_id] += 1

        self._analyze_patterns(pattern_id)
        self._save_patterns()

    def _analyze_patterns(self, pattern_id: str) -> None:
        """
        Enhanced pattern analysis with meta-learning capabilities.
        Adapts analysis based on pattern effectiveness history.
        
        Args:
            pattern_id: Pattern to analyze
        """
        if pattern_id not in self.interaction_patterns:
            return

        pattern_data = self.interaction_patterns[pattern_id]
        if len(pattern_data) < 2:
            return

        # Get historical effectiveness
        historical_effectiveness = self._get_pattern_effectiveness(pattern_id)

        # Analyze timing patterns with adaptive window
        window_size = max(2, min(len(pattern_data),
                               int(5 + (5 * historical_effectiveness))))
        recent_data = pattern_data[-window_size:]

        intervals = []
        for i in range(1, len(recent_data)):
            time_diff = (recent_data[i][0] - recent_data[i-1][0]).total_seconds()
            intervals.append(time_diff)

        if not intervals:
            return

        # Calculate pattern regularity with adaptive weights
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        base_regularity = 1.0 - min(1.0, variance / (avg_interval ** 2))

        # Apply temporal decay to older intervals
        weighted_regularity = self._calculate_temporal_regularity(
            intervals, base_regularity, historical_effectiveness)

        # Enhanced message analysis
        messages = [msg for _, msg, _ in recent_data]
        base_similarity = self._calculate_message_similarity(messages)

        # Apply adaptive content analysis
        content_score = self._analyze_content_evolution(
            messages, historical_effectiveness)

        # Calculate adaptive weights
        timing_weight = 0.4 + (0.2 * historical_effectiveness)
        similarity_weight = 0.3 + (0.1 * (1 - historical_effectiveness))
        evolution_weight = 1 - (timing_weight + similarity_weight)

        # Update pattern score with weighted components
        self.pattern_scores[pattern_id] = (
            (timing_weight * weighted_regularity) +
            (similarity_weight * base_similarity) +
            (evolution_weight * content_score)
        )

        # Record effectiveness for meta-learning
        self._update_effectiveness(pattern_id)

    def _get_pattern_effectiveness(self, pattern_id: str) -> float:
        """Calculate historical pattern recognition effectiveness."""
        if pattern_id not in self.interaction_patterns:
            return 0.5

        pattern_data = self.interaction_patterns[pattern_id]
        if len(pattern_data) < 3:
            return 0.5

        # Calculate prediction accuracy
        scores = [score for _, _, score in pattern_data]
        predictions = scores[:-1]
        actuals = scores[1:]

        accuracy = sum(1 - abs(p - a) for p, a in zip(predictions, actuals))
        accuracy = accuracy / len(predictions)

        return min(1.0, max(0.1, accuracy))

    def _calculate_temporal_regularity(self, intervals: List[float],
                                     base_regularity: float,
                                     effectiveness: float) -> float:
        """Calculate regularity with temporal weighting."""
        if not intervals:
            return base_regularity

        # Apply temporal decay
        weights = [1 - (i / (len(intervals) + 1)) for i in range(len(intervals))]

        # Adjust weights based on effectiveness
        weights = [w * (0.5 + (0.5 * effectiveness)) for w in weights]

        # Calculate weighted regularity
        weighted_sum = sum(w * abs(1 - (i / intervals[0]))
                         for w, i in zip(weights, intervals))
        weighted_regularity = 1 - (weighted_sum / sum(weights))

        return min(1.0, max(0.0, weighted_regularity))

    def _analyze_content_evolution(self, messages: List[str],
                                 effectiveness: float) -> float:
        """Analyze how content evolves over interactions."""
        if len(messages) < 2:
            return 0.5

        # Calculate content evolution metrics
        evolution_scores = []
        for i in range(1, len(messages)):
            prev_set = set(messages[i-1].lower().split())
            curr_set = set(messages[i].lower().split())

            # Measure meaningful changes
            changes = len(prev_set ^ curr_set)  # Symmetric difference
            total_words = len(prev_set | curr_set)

            # Score evolution (changes while maintaining core message)
            evolution = 1 - (changes / max(1, total_words))
            evolution_scores.append(evolution)

        # Weight recent evolution more heavily
        weights = [1 + (i / len(evolution_scores))
                  for i in range(len(evolution_scores))]

        # Adjust weights based on effectiveness
        weights = [w * (0.5 + (0.5 * effectiveness)) for w in weights]

        # Calculate weighted evolution score
        weighted_evolution = (
            sum(w * s for w, s in zip(weights, evolution_scores)) /
            sum(weights)
        )

        return min(1.0, max(0.0, weighted_evolution))

    def _update_effectiveness(self, pattern_id: str) -> None:
        """Update pattern effectiveness metrics for meta-learning."""
        if pattern_id not in self.interaction_patterns:
            return

        pattern_data = self.interaction_patterns[pattern_id]
        if len(pattern_data) < 2:
            return

        # Update latest interaction score based on prediction accuracy
        prediction = self.get_pattern_prediction(pattern_id)
        if prediction:
            latest = pattern_data[-1]
            predicted_time = prediction['predicted_time']
            actual_time = latest[0]

            # Calculate temporal accuracy
            time_diff = abs((actual_time - predicted_time).total_seconds())
            time_accuracy = 1.0 / (1.0 + (time_diff / 3600))  # Hourly scale

            # Update score
            pattern_data[-1] = (latest[0], latest[1], time_accuracy)
            self.interaction_patterns[pattern_id] = pattern_data

    def _calculate_message_similarity(self, messages: List[str]) -> float:
        """Calculate similarity score for a set of messages."""
        if not messages or len(messages) < 2:
            return 0.0

        # Simple word-based similarity for now
        word_sets = [set(msg.lower().split()) for msg in messages]
        base_set = word_sets[0]

        similarities = []
        for other_set in word_sets[1:]:
            intersection = len(base_set & other_set)
            union = len(base_set | other_set)
            similarities.append(intersection / max(1, union))

        return sum(similarities) / len(similarities)

    def get_pattern_prediction(self, pattern_id: str) -> Optional[dict]:
        """
        Get prediction for next pattern occurrence.
        
        Args:
            pattern_id: Pattern to analyze
            
        Returns:
            dict with prediction details or None
        """
        if pattern_id not in self.interaction_patterns:
            return None

        pattern_data = self.interaction_patterns[pattern_id]
        if len(pattern_data) < 2:
            return None

        # Get timing pattern
        last_time = pattern_data[-1][0]
        intervals = []
        for i in range(1, len(pattern_data)):
            time_diff = (pattern_data[i][0] - pattern_data[i-1][0]).total_seconds()
            intervals.append(time_diff)

        if not intervals:
            return None

        # Calculate prediction
        avg_interval = sum(intervals) / len(intervals)
        predicted_next = last_time + timedelta(seconds=avg_interval)

        return {
            'pattern_id': pattern_id,
            'confidence': self.pattern_scores.get(pattern_id, 0.0),
            'predicted_time': predicted_next,
            'avg_interval': avg_interval,
            'last_occurrence': last_time
        }

    def get_agent_patterns(self, agent_id: str) -> Dict[str, float]:
        """Get patterns associated with an agent."""
        return self.agent_patterns.get(agent_id, {})

    def cleanup_old_patterns(self, max_age_days: int = 30) -> None:
        """Clean up old pattern data."""
        now = datetime.now()
        cutoff = now - timedelta(days=max_age_days)

        # Clean up old interactions
        for pattern_id, interactions in list(self.interaction_patterns.items()):
            self.interaction_patterns[pattern_id] = [
                (d, m, s) for d, m, s in interactions
                if d > cutoff
            ]

            # Remove empty patterns
            if not self.interaction_patterns[pattern_id]:
                self.interaction_patterns.pop(pattern_id)
                self.pattern_scores.pop(pattern_id, None)

        # Clean up agent patterns
        for agent_id in list(self.agent_patterns.keys()):
            self.agent_patterns[agent_id] = {
                p: s for p, s in self.agent_patterns[agent_id].items()
                if p in self.interaction_patterns
            }
            if not self.agent_patterns[agent_id]:
                self.agent_patterns.pop(agent_id)

        self._save_patterns()
