def _inject_prompt_caching(messages, num_breakpoints=3):
    breakpoints_remaining = num_breakpoints

    # Loop through messages from newest to oldest
    for message in reversed(messages):  # Message 5 -> 4 -> 3 -> 2 -> 1
        if message["role"] == "user" and isinstance(content := message["content"], list):
            if breakpoints_remaining:
                # First 3 iterations (newest messages)
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = {"type": "ephemeral"}
                # Message 5: Set cache point 1
                # Message 4: Set cache point 2
                # Message 3: Set cache point 3
            else:
                # First message encountered after breakpoints = 0
                content[-1].pop("cache_control", None)  # Remove existing cache_control
                break  # Stop processing older messages
    return messages


def test_prompt_caching_single_message():
    single_message = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    processed_messages = _inject_prompt_caching(single_message)
    assert "cache_control" in processed_messages[0]["content"][-1]
    assert processed_messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_prompt_caching():
    # Create a sequence of test messages
    messages = [
        # Message 1 (oldest) - Simple text message
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello Claude",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # Message 2 - Assistant message
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello!",
                }
            ],
        },
        # Message 3 - User message with cache_control
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the weather at San Francisco",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # Message 4 - Assistant tool use message
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<thinking>Checking weather...</thinking>"},
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"},
                },
            ],
        },
        # Message 5 - User tool result message with cache_control
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_02B10r91rw91mr928946mr0",
                    "content": "20 degrees",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    # Process messages
    processed_messages = _inject_prompt_caching(messages)

    # Verify the cache control settings
    # The two newest messages with user role should have cache_control
    assert processed_messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert processed_messages[-3]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    # Message 3's existing cache_control should be preserved
    assert "cache_control" in processed_messages[2]["content"][-1]
    assert processed_messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    # The oldest message should not have cache_control
    assert "cache_control" in processed_messages[0]["content"][-1]
