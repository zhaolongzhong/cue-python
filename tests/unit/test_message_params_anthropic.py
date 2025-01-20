from anthropic.types import MessageParam as AnthropicMessageParam

# Reference: https://docs.anthropic.com/en/api/messages-examples


def test_user():
    message_dict = {"role": "user", "content": "Hello, Claude"}
    message_param = AnthropicMessageParam(**message_dict)
    assert message_param["role"] == "user"
    assert message_param["content"] == "Hello, Claude"


def test_assistant():
    message_dict = {"role": "assistant", "content": "Hello!"}
    message_param = AnthropicMessageParam(**message_dict)
    assert message_param["role"] == "assistant"
    assert message_param["content"] == "Hello!"


def test_assistant_response():
    message_dict = {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }

    message_param = AnthropicMessageParam(**message_dict)
    assert message_param["role"] == "assistant"
    assert message_param["content"][0]["text"] == "Hello!"


def test_tool_use():
    message_dict = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """<thinking>To answer this question, I will: 1. Use the get_weather tool to get the
                current weather in San Francisco. 2. Use the get_time tool to get the current time in the
                America/Los_Angeles timezone, which covers San Francisco, CA.</thinking>""",
            },
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lq9",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    }
    param = AnthropicMessageParam(**message_dict)
    assert param["role"] == "assistant"
    assert param["content"][0]["type"] == "text"
    assert param["content"][1]["type"] == "tool_use"


def test_tool_result():
    message_dict = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                "content": "15 degrees",
            },
        ],
    }

    param = AnthropicMessageParam(**message_dict)
    assert param["role"] == "user"
    assert param["content"][0]["type"] == "tool_result"
    assert param["content"][0]["tool_use_id"] == "toolu_01A09q90qw90lq917835lq9"
    assert param["content"][0]["content"] == "15 degrees"


def test_tool_result_with_image():
    """Multi tool results including image"""
    message_dict = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                "content": [
                    {"type": "text", "text": "15 degrees"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "/9j/4AAQSkZJRg...",
                        },
                    },
                ],
                "is_error": False,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    param = AnthropicMessageParam(**message_dict)
    assert param["role"] == "user"
    assert param["content"][0]["type"] == "tool_result"
    assert param["content"][0]["tool_use_id"] == "toolu_01A09q90qw90lq917835lq9"
    assert param["content"][0]["content"][0]["type"] == "text"
    assert param["content"][0]["content"][0]["text"] == "15 degrees"
    assert param["content"][0]["content"][1]["type"] == "image"
    assert param["content"][0]["content"][1]["source"]["type"] == "base64"


def test_tool_result_empty():
    message_dict = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
            }
        ],
    }

    param = AnthropicMessageParam(**message_dict)
    assert param["role"] == "user"
    assert param["content"][0]["type"] == "tool_result"
    assert param["content"][0]["tool_use_id"] == "toolu_01A09q90qw90lq917835lq9"
