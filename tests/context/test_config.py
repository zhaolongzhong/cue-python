"""Tests for context configuration."""

import pytest

from cue.schemas import FeatureFlag
from cue.context.config import ContextConfig


def test_valid_config():
    """Test creation of valid configuration."""
    config = ContextConfig(
        model="gpt-4",
        max_tokens=1000,
        feature_flag=FeatureFlag(),
        batch_remove_percentage=0.3,
        max_summaries=6,
    )

    assert config.model == "gpt-4"
    assert config.max_tokens == 1000
    assert config.batch_remove_percentage == 0.3
    assert config.max_summaries == 6


def test_invalid_max_tokens():
    """Test validation of max_tokens."""
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        ContextConfig(
            model="gpt-4",
            max_tokens=0,
            feature_flag=FeatureFlag(),
        )


def test_invalid_batch_percentage():
    """Test validation of batch_remove_percentage."""
    with pytest.raises(ValueError, match="batch_remove_percentage must be between 0 and 1"):
        ContextConfig(
            model="gpt-4",
            max_tokens=1000,
            feature_flag=FeatureFlag(),
            batch_remove_percentage=1.5,
        )


def test_invalid_min_tokens():
    """Test validation of min_tokens_to_keep."""
    with pytest.raises(ValueError, match="min_tokens_to_keep must be less than max_tokens"):
        ContextConfig(
            model="gpt-4",
            max_tokens=1000,
            feature_flag=FeatureFlag(),
            min_tokens_to_keep=1200,
        )
