"""Tests for token management component."""

from unittest.mock import MagicMock

import pytest

from cue.utils import TokenCounter
from cue.schemas import FeatureFlag
from cue.context.config import ContextConfig
from cue.context.token_manager import TokenInfo, TokenManager


@pytest.fixture
def config():
    return ContextConfig(
        model="gpt-4",
        max_tokens=1000,
        feature_flag=FeatureFlag(),
        batch_remove_percentage=0.3,
        min_tokens_to_keep=300,
    )


@pytest.fixture
def counter():
    counter = TokenCounter()
    counter.count_messages_tokens = MagicMock(return_value=100)
    counter.count_dict_tokens = MagicMock(return_value=25)
    return counter


@pytest.fixture
def token_manager(config, counter):
    return TokenManager(config, counter)


def test_analyze_tokens_under_limit(token_manager, counter):
    """Test token analysis when under limit."""
    counter.count_messages_tokens.return_value = 800
    messages = [{"role": "user", "content": "test"}]

    info = token_manager.analyze_tokens(messages)

    assert isinstance(info, TokenInfo)
    assert info.total_tokens == 800
    assert not info.needs_truncation
    assert info.tokens_to_remove is None


def test_analyze_tokens_over_limit(token_manager, counter):
    """Test token analysis when over limit."""
    counter.count_messages_tokens.return_value = 1200
    messages = [{"role": "user", "content": "test"}]

    info = token_manager.analyze_tokens(messages)

    assert info.total_tokens == 1200
    assert info.needs_truncation
    assert info.tokens_to_remove == 500  # To get to 70% of max_tokens


def test_can_add_message_under_limit(token_manager):
    """Test message addition when under token limit."""
    current_tokens = 800
    message_tokens = 100

    assert token_manager.can_add_message(current_tokens, message_tokens)


def test_can_add_message_at_limit(token_manager):
    """Test message addition exactly at token limit."""
    current_tokens = 900
    message_tokens = 100

    assert token_manager.can_add_message(current_tokens, message_tokens)


def test_can_add_message_over_limit(token_manager):
    """Test message addition when it would exceed limit."""
    current_tokens = 950
    message_tokens = 100

    assert not token_manager.can_add_message(current_tokens, message_tokens)


def test_estimate_message_tokens(token_manager, counter):
    """Test token estimation for a single message."""
    message = {"role": "user", "content": "test"}
    counter.count_dict_tokens.return_value = 25

    tokens = token_manager.estimate_message_tokens(message)

    assert tokens == 25
    counter.count_dict_tokens.assert_called_once_with(message)
