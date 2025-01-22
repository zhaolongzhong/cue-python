"""Tests for the pattern learning and self-reflection system."""

import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from src.cue.reflection.pattern_learner import PatternLearner


class TestPatternLearner(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("/tmp/test_pattern_learner")
        self.test_file = self.test_dir / "learning_state.json"

        self.test_dir.mkdir(exist_ok=True)

        self.patcher = patch(
            'src.cue.reflection.pattern_learner.Path.expanduser',
            return_value=str(self.test_dir)
        )
        self.patcher.start()

        self.learner = PatternLearner()

    def tearDown(self):
        """Clean up test environment."""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

        self.patcher.stop()

    def test_learning_curve_analysis(self):
        """Test learning curve analysis."""
        pattern_id = "test_pattern"

        # Simulate improving effectiveness
        now = datetime.now()
        effectiveness_values = [0.5, 0.6, 0.7, 0.8, 0.9]

        for i, effectiveness in enumerate(effectiveness_values):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            self.learner.analyze_pattern(pattern_id, pattern_data, effectiveness)

        # Get learning summary
        summary = self.learner.get_learning_summary(pattern_id)

        self.assertEqual(summary['status'], 'active')
        self.assertEqual(summary['events_recorded'], 5)
        self.assertGreater(summary['learning_rate'], 0)
        self.assertGreater(summary['stability'], 0.8)

    def test_adaptation_analysis(self):
        """Test adaptation rate analysis."""
        pattern_id = "adaptation_test"

        # Simulate pattern with recovery after effectiveness drops
        effectiveness_sequence = [0.8, 0.7, 0.4, 0.6, 0.8]
        now = datetime.now()

        for i, effectiveness in enumerate(effectiveness_sequence):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            result = self.learner.analyze_pattern(
                pattern_id, pattern_data, effectiveness)

        self.assertGreater(result['adaptation_rate'], 0)

        # Verify insight generation
        self.assertIsNotNone(result['latest_insight'])
        self.assertEqual(
            result['latest_insight']['type'],
            'rapid_adaptation'
        )

    def test_insight_generation(self):
        """Test learning insight generation."""
        pattern_id = "insight_test"

        # Simulate rapid improvement
        now = datetime.now()
        effectiveness_values = [0.3, 0.4, 0.7, 0.9]

        insights = []
        for i, effectiveness in enumerate(effectiveness_values):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            result = self.learner.analyze_pattern(
                pattern_id, pattern_data, effectiveness)
            if result['latest_insight']:
                insights.append(result['latest_insight'])

        self.assertTrue(insights)
        self.assertEqual(
            insights[-1]['type'],
            'accelerated_learning'
        )

    def test_learning_persistence(self):
        """Test learning state persistence."""
        pattern_id = "persistence_test"

        # Record some learning events
        now = datetime.now()
        for i in range(3):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            self.learner.analyze_pattern(pattern_id, pattern_data, 0.5 + (i * 0.2))

        # Create new learner instance
        new_learner = PatternLearner()

        # Verify state loaded
        summary = new_learner.get_learning_summary(pattern_id)
        self.assertEqual(summary['status'], 'active')
        self.assertEqual(summary['events_recorded'], 3)

    def test_stability_analysis(self):
        """Test stability measurement."""
        pattern_id = "stability_test"

        # Test stable pattern
        now = datetime.now()
        stable_values = [0.8, 0.79, 0.81, 0.8, 0.82]

        for i, effectiveness in enumerate(stable_values):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            self.learner.analyze_pattern(pattern_id, pattern_data, effectiveness)

        summary = self.learner.get_learning_summary(pattern_id)
        self.assertGreater(summary['stability'], 0.9)

        # Test unstable pattern
        pattern_id = "unstable_test"
        unstable_values = [0.8, 0.4, 0.9, 0.3, 0.85]

        for i, effectiveness in enumerate(unstable_values):
            pattern_data = {"timestamp": now + timedelta(hours=i)}
            self.learner.analyze_pattern(pattern_id, pattern_data, effectiveness)

        summary = self.learner.get_learning_summary(pattern_id)
        self.assertLess(summary['stability'], 0.5)
