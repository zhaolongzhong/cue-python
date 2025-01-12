"""
Tool output formatting utilities for better readability
"""
import re
from enum import Enum
from typing import Any, Dict, List, Union, Optional
from textwrap import indent
from dataclasses import dataclass


class FormatStyle(str, Enum):
    """Output format styles"""
    PLAIN = "plain"  # Simple text output
    STRUCTURED = "structured"  # Formatted with headers and sections
    COMPACT = "compact"  # Minimal formatting
    MARKDOWN = "markdown"  # Markdown formatted


@dataclass
class CodeBlock:
    """Code block with language and content"""
    language: str
    content: str


@dataclass
class FileContent:
    """File content with path and line numbers"""
    path: str
    content: str
    line_range: Optional[tuple[int, int]] = None


@dataclass
class DiffContent:
    """Git diff content"""
    content: str
    file_path: Optional[str] = None
    stats: Optional[Dict[str, int]] = None


class ToolOutputFormatter:
    """Formats tool outputs for better readability"""

    def __init__(self, style: FormatStyle = FormatStyle.STRUCTURED):
        self.style = style

    def format_output(self, output: Any) -> str:
        """Format any tool output based on content type"""
        if isinstance(output, (str, int, float, bool)):
            return str(output)
        elif isinstance(output, (list, tuple)):
            return self._format_list(output)
        elif isinstance(output, dict):
            return self._format_dict(output)
        elif isinstance(output, (FileContent, CodeBlock, DiffContent)):
            return self._format_special_content(output)
        else:
            return str(output)

    def _format_list(self, items: List[Any], indent_level: int = 0) -> str:
        """Format a list of items"""
        if not items:
            return "[]"

        if self.style == FormatStyle.COMPACT:
            return str(items)

        if all(isinstance(x, (str, int, float, bool)) for x in items):
            # Simple items on one line
            return f"[{', '.join(str(x) for x in items)}]"

        # Complex items get their own lines
        result = "[\n"
        for item in items:
            formatted = indent(self.format_output(item), "  " * (indent_level + 1))
            result += f"{formatted},\n"
        result += "  " * indent_level + "]"
        return result

    def _format_dict(self, data: Dict[str, Any], indent_level: int = 0) -> str:
        """Format a dictionary"""
        if not data:
            return "{}"

        if self.style == FormatStyle.COMPACT:
            return str(data)

        result = "{\n"
        for key, value in data.items():
            formatted_value = self.format_output(value, indent_level + 1)
            result += f"{'  ' * (indent_level + 1)}{key}: {formatted_value}\n"
        result += "  " * indent_level + "}"
        return result

    def _format_special_content(self, content: Union[FileContent, CodeBlock, DiffContent]) -> str:
        """Format special content types"""
        if isinstance(content, FileContent):
            return self._format_file_content(content)
        elif isinstance(content, CodeBlock):
            return self._format_code_block(content)
        else:  # DiffContent
            return self._format_diff_content(content)

    def _format_file_content(self, file_content: FileContent) -> str:
        """Format file content with line numbers"""
        if self.style == FormatStyle.COMPACT:
            return f"{file_content.path}:\n{file_content.content}"

        lines = file_content.content.splitlines()
        if not lines:
            return f"File: {file_content.path} (empty)"

        start_line = file_content.line_range[0] if file_content.line_range else 1
        result = [f"File: {file_content.path}"]

        if file_content.line_range:
            result.append(f"Lines: {file_content.line_range[0]}-{file_content.line_range[1]}")

        result.append("-" * 40)

        for i, line in enumerate(lines, start=start_line):
            result.append(f"{i:4d}| {line}")

        return "\n".join(result)

    def _format_code_block(self, code: CodeBlock) -> str:
        """Format code block with syntax highlighting markers"""
        if self.style == FormatStyle.MARKDOWN:
            return f"```{code.language}\n{code.content}\n```"
        else:
            return f"Language: {code.language}\n{'-' * 40}\n{code.content}"

    def _format_diff_content(self, diff: DiffContent) -> str:
        """Format git diff content"""
        result = []
        if diff.file_path:
            result.append(f"Diff for: {diff.file_path}")

        if diff.stats:
            stats = [f"{k}: {v}" for k, v in diff.stats.items()]
            result.append("Changes: " + ", ".join(stats))

        result.append("-" * 40)
        result.append(diff.content)

        return "\n".join(result)

    @staticmethod
    def detect_content_type(text: str) -> Union[FileContent, CodeBlock, DiffContent, str]:
        """Detect the type of content from text"""
        # Check for file content (line numbers)
        if re.match(r'^\s*\d+\|\s', text, re.MULTILINE):
            return FileContent(path="unknown", content=text)

        # Check for git diff
        if text.startswith("diff --git") or re.search(r'^\+\+\+\s+b/', text, re.MULTILINE):
            return DiffContent(content=text)

        # Check for code block
        if re.match(r'^\s*(?:def|class|import|from|#include|package)\s', text, re.MULTILINE):
            return CodeBlock(language="python", content=text)

        return text
