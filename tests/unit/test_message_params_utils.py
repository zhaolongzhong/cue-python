from cue.utils.mesage_params_utils import has_tool_calls, is_tool_result, get_text_from_message_params


def test_has_tool_call():
    message_dict = {"role": "assistant", "content": "Hello!"}
    assert has_tool_calls(msg=message_dict) is False

    message_dict = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """<thinking>To answer this question, I will: 1. Use the get_weather tool to get the current
                weather in San Francisco. 2. Use the get_time tool to get the current time in the America/Los_Angeles
                timezone, which covers San Francisco, CA.</thinking>""",
            },
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lq9",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    }
    assert has_tool_calls(msg=message_dict) is True


def test_is_tool_result():
    message_dict = {"role": "assistant", "content": "Hello!"}
    assert is_tool_result(msg=message_dict) is False

    message_dict = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": """<thinking>To answer this question, I will: 1. Use the get_weather tool to get the current
                weather in San Francisco. 2. Use the get_time tool to get the current time in the America/Los_Angeles
                timezone, which covers San Francisco, CA.</thinking>""",
            },
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lq9",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    }
    assert is_tool_result(msg=message_dict) is False

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
    assert is_tool_result(msg=message_dict) is True, "Should be tool result"


def test_get_text_from_message_params_claude_basic():
    """Test basic Claude message format"""
    messages = [
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]

    result = get_text_from_message_params(model="claude-3", messages=messages)
    assert "Hello, Claude" in result
    assert "Hello! How can I help you?" in result


def test_get_text_from_message_params_claude_tool_use():
    """Test Claude message with tool use"""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"location": "San Francisco"}},
            ],
        }
    ]

    result = get_text_from_message_params(model="claude-3", messages=messages)
    assert "Let me check the weather." in result
    assert "type: tool_use" in result
    assert "name: get_weather" in result


def test_get_text_from_message_params_claude_tool_result():
    """Test Claude message with tool result"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": [{"type": "text", "text": "72°F, Sunny"}]}
            ],
        }
    ]

    result = get_text_from_message_params(model="claude-3", messages=messages)
    assert "72°F, Sunny" in result
    assert "tool_use_id: tool_1" in result


def test_get_text_from_message_params_openai_basic():
    """Test basic OpenAI message format"""
    messages = [
        {"role": "user", "content": "Hello, ChatGPT"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]

    result = get_text_from_message_params(model="gpt-4", messages=messages)
    assert "Hello, ChatGPT" in result
    assert "Hello! How can I help you?" in result


def test_get_text_from_message_params_openai_tool_calls():
    """Test OpenAI message with tool calls"""
    messages = [
        {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "San Francisco"}'},
                }
            ],
        }
    ]

    result = get_text_from_message_params(model="gpt-4", messages=messages)
    assert "Let me check that for you." in result
    assert "type: function" in result
    assert "name: get_weather" in result


def test_get_text_from_message_params_openai_tool_result():
    """Test OpenAI message with tool result"""
    messages = [{"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "72°F, Sunny"}]

    result = get_text_from_message_params(model="gpt-4", messages=messages)
    assert "72°F, Sunny" in result
    assert "tool_call_id: call_1" in result


def test_get_text_from_message_params_with_image():
    """Test handling of image content"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
    ]

    result = get_text_from_message_params(model="claude-3", messages=messages)
    assert "What's in this image?" in result
    assert "skip image content" in result
