from typing import Optional

from ..types import MessageParam
from .message import Message


class MessageParamFactory:
    @staticmethod
    def from_message(
        message: Message,
        force_str_content: bool = False,
        truncate_length: Optional[int] = None,
        truncation_indicator: str = " ...",
        show_visibility: bool = True,
    ) -> MessageParam:
        """
        Create a MessageParam instance from a Message.

        Args:
            message: The Message instance to convert from.
            force_str_content: If True, converts content to string format.
            truncate_length: Maximum length to truncate content to, if provided.
            truncation_indicator: String to append when content is truncated
            show_visibility: If True, includes visibility percentage in truncation indicator
        """
        role = message.author.role
        content = message.content.content
        if force_str_content:
            content = str(message.content.content)
            original_length = len(content)

            if truncate_length is not None and original_length > truncate_length:
                # Calculate visibility based on truncate_length
                visibility_percent = round((truncate_length / original_length) * 100)

                # Create indicator with visibility
                if show_visibility:
                    indicator = f"{truncation_indicator} ({visibility_percent}% visible)"
                else:
                    indicator = truncation_indicator

                # Calculate actual content length
                actual_length = truncate_length - len(indicator)
                content = content[:actual_length] + indicator

            if "tool" == message.author.role:
                role = "assistant"

        return MessageParam(
            role=role,
            content=content,
            name=message.author.name if hasattr(message.author, "name") else None,
            msg_id=message.id,
        )
