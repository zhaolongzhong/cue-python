"""Unit tests for the DynamicMemoryManager class."""

import pytest

from cue.utils.token_counter import TokenCounter
from cue.memory.memory_manager import DynamicMemoryManager

pytestmark = pytest.mark.unit


@pytest.fixture
def memory_manager():
    """Create a DynamicMemoryManager instance for testing."""
    return DynamicMemoryManager(max_tokens=100, max_chars=50)


@pytest.fixture
def sample_memories():
    """Create a sample set of memories for testing."""
    return {
        "memory1": "First test memory",
        "memory2": "Second test memory",
        "memory3": "Third test memory",
    }


def test_memory_initialization(memory_manager):
    """Test that memory manager initializes with correct settings."""
    assert memory_manager.max_tokens == 100
    assert memory_manager.max_chars == 50
    assert isinstance(memory_manager.token_counter, TokenCounter)
    assert len(memory_manager.memories) == 0
    assert memory_manager.recent_memories is None


def test_add_single_memory(memory_manager):
    """Test adding a single memory."""
    memories = {"mem1": "Test memory content"}
    memory_manager.add_memories(memories)

    assert len(memory_manager.memories) == 1
    assert "mem1" in memory_manager.memories
    assert memory_manager.memories["mem1"] == "Test memory content"


def test_memory_truncation(memory_manager):
    """Test that long memories are truncated correctly."""
    long_memory = "This is a very long memory that should be truncated in the middle of the content"
    memories = {"long_mem": long_memory}

    memory_manager.add_memories(memories)
    truncated = memory_manager.memories["long_mem"]

    assert len(truncated) <= memory_manager.max_chars
    assert "..." in truncated
    assert truncated.startswith("This")
    assert truncated.endswith("content")


def test_token_limit_enforcement(memory_manager):
    """Test that token limits are enforced when adding memories."""
    # Create memories that will exceed token limit
    long_memories = {
        f"mem{i}": f"Memory content {i} with enough text to consume tokens {i * 'extra '}" for i in range(10)
    }

    memory_manager.add_memories(long_memories)
    stats = memory_manager.get_memory_stats()

    assert stats["total_tokens"] <= memory_manager.max_tokens
    assert len(memory_manager.memories) < len(long_memories)


def test_clear_memories(memory_manager, sample_memories):
    """Test clearing all memories."""
    memory_manager.add_memories(sample_memories)
    assert len(memory_manager.memories) > 0

    memory_manager.clear_memories()
    assert len(memory_manager.memories) == 0
    assert memory_manager.get_formatted_memories() is None


def test_memory_formatting(memory_manager, sample_memories):
    """Test memory formatting for LLM consumption."""
    memory_manager.add_memories(sample_memories)
    formatted = memory_manager.get_formatted_memories()

    assert formatted is not None
    assert "<recent_memories>" in formatted
    assert "</recent_memories>" in formatted
    assert "First test memory" in formatted
    assert "Second test memory" in formatted
    assert "Third test memory" in formatted


def test_memory_stats(memory_manager, sample_memories):
    """Test memory statistics calculation."""
    memory_manager.add_memories(sample_memories)
    stats = memory_manager.get_memory_stats()

    assert isinstance(stats, dict)
    assert "memory_count" in stats
    assert "total_tokens" in stats
    assert "remaining_tokens" in stats
    assert "is_at_capacity" in stats
    assert stats["memory_count"] == len(sample_memories)


def test_empty_memories_handling(memory_manager):
    """Test handling of empty memories."""
    assert memory_manager.get_formatted_memories() is None
    assert memory_manager.get_memory_stats()["memory_count"] == 0
    assert memory_manager.get_memory_stats()["total_tokens"] == 0

    # Adding empty memories should not cause errors
    memory_manager.add_memories({})
    assert len(memory_manager.memories) == 0
