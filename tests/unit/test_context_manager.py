import logging
from typing import Optional

import pytest

from cue.utils.mesage_params_utils import has_tool_calls, is_tool_result
from cue.context.context_window_manager import ContextWindowManager


@pytest.fixture
def context_manager():
    return ContextWindowManager(model="gpt-4o-mini", max_tokens=1000, batch_remove_percentage=0.25)


def create_message(role: str, content: str, has_tool_calls: bool = False, tool_call_id: Optional[str] = None) -> dict:
    """Helper function to create test messages"""
    message = {"role": role, "content": content}
    if has_tool_calls:
        message["tool_calls"] = [{"id": tool_call_id, "type": "function", "function": {"name": "test_tool"}}]
    return message


def create_tool_result(content: str, tool_call_id: str) -> dict:
    """Helper function to create tool result messages"""
    return {
        "tool_call_id": tool_call_id,
        "role": "tool",
        "content": content,
    }


class TestDynamicContextManager:
    def test_initialization(self, context_manager):
        """Test proper initialization of DynamicContextManager"""
        assert context_manager.max_tokens == 1000
        assert context_manager._get_batch_remove_size() == 250  # 25% of max_tokens
        assert len(context_manager.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_batch_message_removal(self, context_manager, monkeypatch):
        """Test batch removal of messages when token limit is exceeded"""

        # Mock token counting to return fixed values
        def mock_count_dict_tokens(data):
            return 200 if data.get("role") == "system" else 100

        monkeypatch.setattr(context_manager.token_counter, "count_dict_tokens", mock_count_dict_tokens)
        monkeypatch.setattr(
            context_manager.token_counter,
            "count_messages_tokens",
            lambda x: sum(mock_count_dict_tokens(msg) for msg in x),
        )

        # Add messages of different priorities
        messages = [
            create_message("system", "System message"),  # HIGH - 200 tokens
            create_message("user", "Query 1"),  # MEDIUM - 100 tokens
            create_message("assistant", "Short"),  # LOW - 100 tokens
            create_message("user", "Query 2"),  # MEDIUM - 100 tokens
            create_message("assistant", "A" * 201),  # MEDIUM - 100 tokens
            create_message("assistant", "OK"),  # LOW - 100 tokens
        ]

        # Add messages one by one and log token counts
        for msg in messages:
            current_tokens = context_manager._get_total_tokens()
            logging.info(f"Before adding message - Total tokens: {current_tokens}")
            await context_manager.add_messages([msg])
            logging.info(f"Added message - Role: {msg.get('role')}, Tokens: {mock_count_dict_tokens(msg)}")

        # Total tokens: 700 (below limit)
        current_count = len(context_manager.get_messages())
        logging.info(f"Current message count: {current_count}")
        assert current_count == 6

        # Add messages that will trigger batch removal
        tool_call_id_2 = "call_2"
        tool_sequence = [
            create_message("assistant", "Using tool", has_tool_calls=True, tool_call_id=tool_call_id_2),  # HIGH
            create_tool_result("Tool result", tool_call_id_2),  # HIGH
        ]

        await context_manager.add_messages(tool_sequence)
        for msg in tool_sequence:
            current_tokens = context_manager._get_total_tokens()
            logging.info(f"Before adding tool message - Total tokens: {current_tokens}")
            logging.info(f"Added tool message - Role: {msg.get('role')}")

    @pytest.mark.asyncio
    async def test_context_stats(self, context_manager, monkeypatch):
        """Test context statistics reporting"""

        def mock_count_dict_tokens(data):
            return 100

        monkeypatch.setattr(context_manager.token_counter, "count_dict_tokens", mock_count_dict_tokens)
        monkeypatch.setattr(context_manager.token_counter, "count_messages_tokens", lambda x: len(x) * 100)

        await context_manager.add_messages([create_message("user", "Test message")])

        stats = context_manager.get_context_stats()
        assert stats["message_count"] == 1
        assert stats["total_tokens"] == 100
        assert stats["remaining_tokens"] == 900
        assert not stats["is_at_capacity"]

    @pytest.mark.asyncio
    async def test_tool_sequence_preservation(self, context_manager, monkeypatch):
        """Test that tool call sequences are preserved during batch removal"""

        def mock_count_dict_tokens(data):
            return 200

        monkeypatch.setattr(context_manager.token_counter, "count_dict_tokens", mock_count_dict_tokens)
        monkeypatch.setattr(context_manager.token_counter, "count_messages_tokens", lambda x: len(x) * 200)

        # Add a tool sequence
        tool_call_id_1 = "call_1"
        tool_sequence = [
            create_message("assistant", "Using tool", has_tool_calls=True, tool_call_id=tool_call_id_1),
            create_tool_result("Tool result", tool_call_id_1),
        ]

        await context_manager.add_messages(tool_sequence)

        # Add more messages to trigger batch removal
        for i in range(5):
            await context_manager.add_messages([create_message("user", f"Message {i}")])
            logging.info(f"Added message {i}")

        # Verify tool sequence is preserved
        messages = context_manager.get_messages()
        tool_call_indices = [i for i, msg in enumerate(messages) if has_tool_calls(msg)]

        for idx in tool_call_indices:
            if idx + 1 < len(messages):
                next_msg = messages[idx + 1]
                assert is_tool_result(next_msg), "Tool call should be followed by tool result"

    @pytest.mark.asyncio
    async def test_tool_sequence_id_matching(self, context_manager, monkeypatch):
        """Test that tool sequences are properly matched and handled using tool_call_ids"""

        def mock_count_dict_tokens(data):
            return 200

        monkeypatch.setattr(context_manager.token_counter, "count_dict_tokens", mock_count_dict_tokens)
        monkeypatch.setattr(context_manager.token_counter, "count_messages_tokens", lambda x: len(x) * 200)

        # Create tool calls message
        tool_calls_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "check_weather", "arguments": '{"city":"New York"}'},
                    "type": "function",
                },
                {
                    "id": "call_2",
                    "function": {"name": "check_weather", "arguments": '{"city":"London"}'},
                    "type": "function",
                },
            ],
        }
        tool_calls_msg_2 = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "function": {"name": "check_weather", "arguments": '{"city":"Tokyo"}'},
                    "type": "function",
                },
            ],
        }
        tool_calls_msg_3 = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_4",
                    "function": {"name": "check_weather", "arguments": '{"city":"Paris"}'},
                    "type": "function",
                },
            ],
        }

        # Create corresponding tool results
        tool_result_1 = {"role": "tool", "content": '{"temperature": "22°C"}', "tool_call_id": "call_1"}
        tool_result_2 = {"role": "tool", "content": '{"temperature": "15°C"}', "tool_call_id": "call_2"}
        tool_result_3 = {"role": "tool", "content": '{"temperature": "25°C"}', "tool_call_id": "call_3"}
        tool_result_4 = {"role": "tool", "content": '{"temperature": "18°C"}', "tool_call_id": "call_4"}

        # Test Case 1: Normal sequence preservation
        messages = [
            {"role": "system", "content": "System message"},
            tool_calls_msg,
            tool_result_1,
            tool_result_2,
            {"role": "user", "content": "Query 1"},
            tool_calls_msg_2,
            tool_result_3,
        ]

        await context_manager.add_messages(messages)
        # Verify initial integrity
        self._verify_tool_sequence_integrity(context_manager.get_messages())

        # Test Case 2: Force removal of older tool sequences
        # Add more messages to force removal of older sequences
        additional_messages = [
            {"role": "user", "content": "Query 2"},
            tool_calls_msg_3,
            tool_result_4,
            {"role": "assistant", "content": "Final reply"},
        ]

        await context_manager.add_messages(additional_messages)
        for msg in additional_messages:
            logging.info(f"Added additional message - Role: {msg.get('role')}, Tool call ID: {msg.get('tool_call_id')}")

        # Verify that sequences are still intact, even if some were removed
        remaining_messages = context_manager.get_messages()
        self._verify_tool_sequence_integrity(remaining_messages)

        # Verify that older sequences were removed completely (not partially)
        tool_call_ids_present = set()
        tool_result_ids_present = set()

        for msg in remaining_messages:
            if msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    tool_call_ids_present.add(call["id"])
            if msg.get("tool_call_id"):
                tool_result_ids_present.add(msg["tool_call_id"])

        # Verify that all present tool calls have their results and vice versa
        assert tool_call_ids_present == tool_result_ids_present, (
            f"Mismatch between tool calls {tool_call_ids_present} and results {tool_result_ids_present}"
        )

        # Test Case 3: Force removal of all tool sequences
        # Add many messages to force removal of all tool sequences
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"Forced removal message {i}"})
        await context_manager.add_messages(msgs)

        final_messages = context_manager.get_messages()

        # Count tool-related messages
        tool_call_count = sum(1 for msg in final_messages if msg.get("tool_calls"))
        tool_result_count = sum(1 for msg in final_messages if msg.get("role") == "tool")

        # Either all tool sequences should be removed, or any remaining sequences should be complete
        if tool_call_count > 0 or tool_result_count > 0:
            self._verify_tool_sequence_integrity(final_messages)

        logging.info("test_tool_sequence_id_matching - Final message sequence:")
        for msg in final_messages:
            logging.info(
                f"Message - Role: {msg.get('role')}, "
                f"Content: {msg.get('content')}, "
                f"Tool calls: {msg.get('tool_calls')}, "
                f"Tool call ID: {msg.get('tool_call_id')}"
            )

    def _verify_tool_sequence_integrity(self, messages):
        """Helper method to verify tool sequence integrity"""
        tool_calls = None
        tool_results = set()

        for msg in messages:
            if msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls")
                # Check that if we have tool calls, we must have all their results
                for call in tool_calls:
                    found_result = False
                    for other_msg in messages:
                        if other_msg.get("role") == "tool" and other_msg.get("tool_call_id") == call["id"]:
                            found_result = True
                            break
                    assert found_result, f"Missing tool result for call {call['id']}"

            if msg.get("role") == "tool":
                # Check that each tool result has its corresponding tool call
                tool_call_id = msg.get("tool_call_id")
                assert tool_call_id, "Tool result must have tool_call_id"
                tool_results.add(tool_call_id)
                found_call = False
                for other_msg in messages:
                    if other_msg.get("tool_calls"):
                        for call in other_msg.get("tool_calls", []):
                            if call["id"] == tool_call_id:
                                found_call = True
                                break
                assert found_call, f"Found orphaned tool result for {tool_call_id}"
