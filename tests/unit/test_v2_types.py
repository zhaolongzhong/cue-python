from cue.v2.types import Tool, Message, InputItem, RunResult, SimpleAgent


def test_simple_agent_creation():
    """Test SimpleAgent can be created with minimal config"""
    agent = SimpleAgent(model="gpt-4o-mini")

    assert agent.model == "gpt-4o-mini"
    assert agent.system_prompt == ""
    assert agent.messages == []
    assert agent.tools == []
    assert agent.max_turns == 10


def test_simple_agent_with_config():
    """Test SimpleAgent with full configuration"""
    agent = SimpleAgent(
        model="claude-3-5-haiku-20241022", system_prompt="You are helpful", max_turns=5, api_key="test-key"
    )

    assert agent.model == "claude-3-5-haiku-20241022"
    assert agent.system_prompt == "You are helpful"
    assert agent.max_turns == 5
    assert agent.api_key == "test-key"


def test_input_item():
    """Test InputItem creation"""
    item = InputItem(type="text", content="Hello world")

    assert item.type == "text"
    assert item.content == "Hello world"


def test_run_result():
    """Test RunResult creation"""
    result = RunResult(content="Response text", usage={"tokens": 100}, metadata={"model": "gpt-4"})

    assert result.content == "Response text"
    assert result.usage["tokens"] == 100
    assert result.metadata["model"] == "gpt-4"


def test_message():
    """Test Message creation"""
    msg = Message(role="user", content="Hello")

    assert msg.role == "user"
    assert msg.content == "Hello"


def test_tool():
    """Test Tool creation"""
    tool = Tool(name="bash", description="Run bash commands", parameters={"type": "object", "properties": {}})

    assert tool.name == "bash"
    assert tool.description == "Run bash commands"
    assert tool.parameters["type"] == "object"
