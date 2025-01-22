"""Tests for the pattern-aware coordination system."""

import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from src.cue.coordination.pattern_coordinator import PatternCoordinator


class TestPatternCoordinator(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Use temporary test directory
        self.test_dir = Path("/tmp/test_pattern_coordinator")
        self.test_file = self.test_dir / "patterns.json"

        # Create test directory
        self.test_dir.mkdir(exist_ok=True)

        # Mock cache directory path
        self.patcher = patch(
            'src.cue.coordination.pattern_coordinator.Path.expanduser',
            return_value=str(self.test_dir)
        )
        self.patcher.start()

        self.coordinator = PatternCoordinator()

    def tearDown(self):
        """Clean up test environment."""
        # Remove test data
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

        self.patcher.stop()

    def test_pattern_detection(self):
        """Test basic pattern detection."""
        pattern_id = "test_pattern"
        message = "Test automation message"

        # Record pattern multiple times with consistent interval
        now = datetime.now()
        with patch('src.cue.coordination.pattern_coordinator.datetime') as mock_dt:
            mock_dt.now.return_value = now
            self.coordinator.record_interaction(pattern_id, message)

            # Second interaction after 10 minutes
            mock_dt.now.return_value = now + timedelta(minutes=10)
            self.coordinator.record_interaction(pattern_id, message)

            # Third interaction after another 10 minutes
            mock_dt.now.return_value = now + timedelta(minutes=20)
            self.coordinator.record_interaction(pattern_id, message)

        # Get prediction
        prediction = self.coordinator.get_pattern_prediction(pattern_id)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction['pattern_id'], pattern_id)
        self.assertGreater(prediction['confidence'], 0.8)  # High confidence due to regularity
        self.assertAlmostEqual(prediction['avg_interval'], 600, delta=1)  # 10 minutes = 600 seconds

    def test_multi_agent_patterns(self):
        """Test multi-agent pattern tracking."""
        pattern_id = "multi_agent_pattern"
        agent_id = "test_agent"
        message = "Multi-agent test message"

        # Record pattern with agent
        self.coordinator.record_interaction(pattern_id, message, agent_id)
        self.coordinator.record_interaction(pattern_id, message, agent_id)

        # Check agent patterns
        agent_patterns = self.coordinator.get_agent_patterns(agent_id)

        self.assertIn(pattern_id, agent_patterns)
        self.assertEqual(agent_patterns[pattern_id], 2)

    def test_pattern_persistence(self):
        """Test pattern data persistence."""
        pattern_id = "persistent_pattern"
        message = "Test persistence"

        # Record pattern
        self.coordinator.record_interaction(pattern_id, message)

        # Create new coordinator instance
        new_coordinator = PatternCoordinator()

        # Verify pattern loaded
        self.assertIn(pattern_id, new_coordinator.interaction_patterns)
        self.assertEqual(
            new_coordinator.interaction_patterns[pattern_id][0][1],
            message
        )

    def test_cleanup(self):
        """Test pattern cleanup."""
        pattern_id = "old_pattern"
        message = "Test cleanup"

        # Record old pattern
        now = datetime.now()
        with patch('src.cue.coordination.pattern_coordinator.datetime') as mock_dt:
            # Record pattern 31 days ago
            mock_dt.now.return_value = now - timedelta(days=31)
            self.coordinator.record_interaction(pattern_id, message)

            # Return to present for cleanup
            mock_dt.now.return_value = now
            self.coordinator.cleanup_old_patterns(max_age_days=30)

        # Pattern should be cleaned up
        self.assertNotIn(pattern_id, self.coordinator.interaction_patterns)

    def test_pattern_analysis(self):
        """Test pattern analysis capabilities."""
        pattern_id = "analysis_pattern"
        base_message = "Test pattern analysis"

        # Record similar but slightly different messages
        messages = [
            base_message,
            base_message + " with small change",
            base_message + " with another change"
        ]

        now = datetime.now()
        with patch('src.cue.coordination.pattern_coordinator.datetime') as mock_dt:
            for i, msg in enumerate(messages):
                mock_dt.now.return_value = now + timedelta(minutes=10 * i)
                self.coordinator.record_interaction(pattern_id, msg)

        # Check pattern score
        self.assertIn(pattern_id, self.coordinator.pattern_scores)
        score = self.coordinator.pattern_scores[pattern_id]

        # Score should be good but not perfect due to message variations
        self.assertGreater(score, 0.6)
        self.assertLess(score, 1.0)
