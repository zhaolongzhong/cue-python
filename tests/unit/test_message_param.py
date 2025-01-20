from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import pytest

from cue.schemas import MessageParamFactory


@dataclass
class MockAuthor:
    role: str
    name: Optional[str] = None


@dataclass
class MockContent:
    content: any


@dataclass
class MockMessage:
    """Mock Message class for testing"""

    id: str
    author: MockAuthor
    content: MockContent
    created_at: datetime = datetime.now()


class TestMessageParam:
    """Test suite for MessageParam class"""

    def create_test_message(
        self, content: any, role: str = "user", name: Optional[str] = None, id: str = "test_id"
    ) -> MockMessage:
        """Helper method to create test messages"""
        return MockMessage(id=id, author=MockAuthor(role=role, name=name), content=MockContent(content=content))

    def test_basic_conversion(self):
        """Test basic message conversion without any special options"""
        message = self.create_test_message(content="Hello, world!")
        param = MessageParamFactory.from_message(message=message)

        assert param.role == "user"
        assert param.content == "Hello, world!"
        assert param.msg_id == "test_id"
        assert param.name is None

    def test_force_str_content(self):
        """Test force_str_content conversion with dict content"""
        dict_content = {"type": "text", "value": "Hello"}
        message = self.create_test_message(content=dict_content)

        param = MessageParamFactory.from_message(message=message, force_str_content=True)
        assert isinstance(param.content, str)
        assert "type" in param.content
        assert "Hello" in param.content

    def test_truncation_basic(self):
        """Test basic content truncation"""
        long_content = "This is a very long message that should be truncated"
        message = self.create_test_message(content=long_content)

        truncate_length = 20
        param = MessageParamFactory.from_message(
            message=message, force_str_content=True, truncate_length=truncate_length, show_visibility=False
        )

        assert len(param.content) <= truncate_length
        assert param.content.endswith("...")

    def test_truncation_with_visibility(self):
        """Test truncation with visibility percentage"""
        content = "This is a message that will be truncated with visibility info"
        message = self.create_test_message(content=content)

        param = MessageParamFactory.from_message(
            message=message, force_str_content=True, truncate_length=25, show_visibility=True
        )

        assert "% visible" in param.content
        assert len(param.content) <= 25

    def test_tool_role_conversion(self):
        """Test tool role conversion to assistant"""
        message = self.create_test_message(content="Tool result", role="tool")
        param = MessageParamFactory.from_message(message=message, force_str_content=True)

        assert param.role == "assistant"

    def test_custom_truncation_indicator(self):
        """Test custom truncation indicator"""
        content = "This is a message that will be truncated"
        message = self.create_test_message(content=content)

        custom_indicator = " (...more)"
        param = MessageParamFactory.from_message(
            message=message,
            force_str_content=True,
            truncate_length=20,
            truncation_indicator=custom_indicator,
            show_visibility=False,
        )

        assert param.content.endswith(custom_indicator)

    def test_author_with_name(self):
        """Test handling of author name"""
        message = self.create_test_message(content="Hello", name="TestUser")
        param = MessageParamFactory.from_message(message=message)

        assert param.name == "TestUser"

    @pytest.mark.parametrize(
        "content,truncate_length,expected_visibility",
        [
            ("12345678910", 5, "45"),  # 5/11 ≈ 45%
            ("12345", 4, "80"),  # 4/5 = 80%
            ("123", 2, "67"),  # 2/3 ≈ 67%
        ],
    )
    def test_visibility_percentage_calculation(self, content: str, truncate_length: int, expected_visibility: str):
        """Test different visibility percentage calculations"""
        message = self.create_test_message(content=content)

        param = MessageParamFactory.from_message(
            message=message, force_str_content=True, truncate_length=truncate_length, show_visibility=True
        )

        assert f"{expected_visibility}% visible" in param.content
