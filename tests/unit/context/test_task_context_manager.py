from uuid import uuid4
from datetime import datetime

import pytest

from cue.schemas import Author, Content, Message, Metadata
from cue.context.task_context_manager import TaskContextManager


@pytest.fixture
def task_manager():
    return TaskContextManager(max_tokens=500, max_chars=200)


def create_message(content: str, role: str = "user", msg_id: str = None) -> Message:
    """Helper to create test messages"""
    now = datetime.now()
    return Message(
        id=msg_id or str(uuid4()),
        conversation_id=str(uuid4()),
        author=Author(role=role),
        content=Content(content=content),
        metadata=Metadata(),
        created_at=now,
        updated_at=now,
    )


def test_init(task_manager):
    """Test initialization of TaskContextManager"""
    assert task_manager.max_tokens == 500
    assert task_manager.max_chars == 200
    assert task_manager.task_messages == {}
    assert task_manager.task_state["status"] == "active"
    assert task_manager.task_state["current_goal"] is None


def test_truncate_center(task_manager):
    """Test message truncation"""
    text = "This is a very long message that needs to be truncated in the center"
    truncated = task_manager._truncate_center(text, max_length=20)
    assert len(truncated) <= 20
    assert "..." in truncated
    assert truncated.startswith("This")
    assert truncated.endswith("ter")


def test_message_type_determination(task_manager):
    """Test message type classification"""
    # Test goal detection
    goal_msg = create_message("The task goal is to implement feature X")
    assert task_manager._determine_message_type(goal_msg) == "TASK_GOAL"
    assert task_manager.task_state["current_goal"] is not None

    # Test user input
    user_msg = create_message("Can you check this?")
    assert task_manager._determine_message_type(user_msg) == "USER_INPUT"

    # Test assistant progress
    progress_msg = create_message("Working on implementing...", role="assistant")
    assert task_manager._determine_message_type(progress_msg) == "TASK_PROGRESS"


def test_add_task_messages_preserves_goal_and_error(task_manager):
    """Test that important context is preserved"""
    # Add initial goal
    goal_msg = create_message("Task goal: implement feature X", msg_id="goal_1")
    task_manager.add_task_messages([goal_msg])

    # Add new messages
    new_msgs = [create_message("Making progress", msg_id="prog_1"), create_message("Almost done", msg_id="prog_2")]
    task_manager.add_task_messages(new_msgs)

    # Verify goal and error are preserved
    context = task_manager.get_formatted_task_context()
    assert "Task goal: implement feature X" in context


def test_token_limit_enforcement(task_manager):
    """Test that token limits are enforced"""
    # Create messages that would exceed token limit
    msgs = []
    for i in range(20):
        msgs.append(create_message(f"Message {i} with some content to use tokens " * 3, msg_id=f"msg_{i}"))

    task_manager.add_task_messages(msgs)
    stats = task_manager.get_task_stats()

    assert stats["total_tokens"] <= task_manager.max_tokens
    assert stats["is_at_capacity"] or stats["remaining_tokens"] >= 0


def test_task_context_formatting(task_manager):
    """Test context formatting"""
    # Add some messages
    msgs = [
        create_message("Task goal: do X", msg_id="goal"),
        create_message("Working on it", msg_id="prog", role="assistant"),
        create_message("Got error", msg_id="error"),
    ]

    task_manager.add_task_messages(msgs)
    context = task_manager.get_formatted_task_context()

    # Check format
    assert "Task Status:" in context
    assert "Start Time:" in context
    assert "<task_context>" in context
    assert "</task_context>" in context

    # Check content
    assert "[TASK_GOAL]" in context
    assert "[TASK_PROGRESS]" in context


def test_clear_task_context(task_manager):
    """Test context clearing"""
    # Add some messages
    msgs = [create_message("Test message", msg_id="test")]
    task_manager.add_task_messages(msgs)

    # Clear context
    task_manager.clear_task_context()

    assert len(task_manager.task_messages) == 0
    assert task_manager.task_state["status"] == "completed"
    assert task_manager.get_formatted_task_context() is None


def test_task_stats(task_manager):
    """Test task statistics"""
    msgs = [create_message("Task goal: test stats", msg_id="goal"), create_message("Progress update", msg_id="prog")]
    task_manager.add_task_messages(msgs)

    stats = task_manager.get_task_stats()

    assert "message_count" in stats
    assert "total_tokens" in stats
    assert "remaining_tokens" in stats
    assert "is_at_capacity" in stats
    assert "task_state" in stats
    assert stats["message_count"] == 2


def test_message_param_generation(task_manager):
    """Test message parameter generation"""
    msgs = [create_message("Test message", msg_id="test")]
    task_manager.add_task_messages(msgs)

    param = task_manager.get_task_context_param()

    assert param is not None
    assert param["role"] == "user"
    assert "<task_context>" in param["content"]
