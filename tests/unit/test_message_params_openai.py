from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
)


def test_user():
    message_dict = {"role": "user", "content": "Hello, ChatGPT"}
    message_param = ChatCompletionUserMessageParam(**message_dict)
    assert message_param["role"] == "user"
    assert message_param["content"] == "Hello, ChatGPT"


def test_user_content():
    message_dict = {
        "role": "user",
        "content": [{"type": "text", "text": "knock knock."}],
    }

    message_param = ChatCompletionUserMessageParam(**message_dict)
    assert message_param["role"] == "user"
    assert message_param["content"][0]["type"] == "text"
    assert message_param["content"][0]["text"] == "knock knock."


def test_assistant():
    message_dict = {"role": "assistant", "content": "Hello!"}
    message_param = ChatCompletionAssistantMessageParam(**message_dict)
    assert message_param["role"] == "assistant"
    assert message_param["content"] == "Hello!"


def test_assistant_dict():
    message_dict = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Who's there?"},
        ],
    }
    message_param = ChatCompletionAssistantMessageParam(**message_dict)
    assert message_param["role"] == "assistant"
    assert message_param["content"][0]["type"] == "text"
    assert message_param["content"][0]["text"] == "Who's there?"


def test_tool_calls():
    message_dict = {
        "content": None,
        "refusal": None,
        "role": "assistant",
        "audio": None,
        "function_call": None,
        "tool_calls": [
            {
                "id": "call_4d7b",
                "function": {
                    "arguments": '{"command":"recall","query":"GitHub collaboration","limit":5}',
                    "name": "memory",
                },
                "type": "function",
            }
        ],
    }
    message_param = ChatCompletionToolParam(**message_dict)
    assert message_param["role"] == "assistant"
    assert message_param["tool_calls"][0]["id"] == "call_4d7b"
    assert message_param["tool_calls"][0]["type"] == "function"


def test_tool_message():
    message_dict = {
        "tool_call_id": "call_4d7b",
        "name": "memory",
        "role": "tool",
        "content": "example content",
    }
    message_param = ChatCompletionToolMessageParam(**message_dict)
    assert message_param["role"] == "tool"
    assert message_param["tool_call_id"] == "call_4d7b"
    assert message_param["content"] == "example content"


def test_tool_result_with_image():
    message_dict = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
        ],
    }

    param = ChatCompletionContentPartImageParam(**message_dict)
    assert param["role"] == "user"
    assert param["content"][0]["type"] == "text"
    assert param["content"][1]["type"] == "image_url"
    assert (
        param["content"][1]["image_url"]["url"]
        == "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
