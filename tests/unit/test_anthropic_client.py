import logging

import pytest

from cue.llm import ChatModel, AnthropicClient
from cue.schemas import AgentConfig

logger = logging.getLogger(__name__)


class TestAnthropicClient:
    @pytest.fixture
    def client(self):
        config = AgentConfig(api_key="dummy_key", model=ChatModel.CLAUDE_3_OPUS_20240229)
        return AnthropicClient(config)

    def test_process_messages_with_name(self, client: AnthropicClient):
        input_messages = [
            {"role": "system", "name": "agent_b", "content": "Your name is agent_b"},
            {"role": "assistant", "name": "agent_a", "content": "Hello, world!"},
        ]
        message_from = client.message_from_template.format(agent_id="agent_a")
        expected_output = [
            {"role": "system", "content": "Your name is agent_b"},
            {"role": "user", "content": f"{message_from} Hello, world!"},
        ]
        result = client._process_messages(input_messages)
        assert result == expected_output

    def test_process_messages_without_name(self, client: AnthropicClient):
        input_messages = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "It's sunny today."},
        ]
        expected_output = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "user", "content": "It's sunny today."},
        ]
        result = client._process_messages(input_messages)
        assert result == expected_output

    def test_process_mixed_messages(self, client: AnthropicClient):
        input_messages = [
            {"role": "system", "name": "agent_b", "content": "Your name is agent_b"},
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "name": "agent_a", "content": "My name is agent_a."},
        ]
        message_from = client.message_from_template.format(agent_id="agent_a")
        expected_output = [
            {"role": "system", "content": "Your name is agent_b"},
            {"role": "user", "content": "What's your name?"},
            {"role": "user", "content": f"{message_from} My name is agent_a."},
        ]
        result = client._process_messages(input_messages)
        assert result == expected_output


if __name__ == "__main__":
    pytest.main()
