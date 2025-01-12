"""Tests for the tool output formatter"""
from textwrap import dedent

from cue.schemas.tool_output_formatter import CodeBlock, DiffContent, FileContent, FormatStyle, ToolOutputFormatter


def test_format_simple_types():
    formatter = ToolOutputFormatter()

    assert formatter.format_output("test") == "test"
    assert formatter.format_output(123) == "123"
    assert formatter.format_output(True) == "True"
    assert formatter.format_output(3.14) == "3.14"


def test_format_list():
    formatter = ToolOutputFormatter()

    # Simple list
    assert formatter.format_output([1, 2, 3]) == "[1, 2, 3]"

    # Nested list
    complex_list = [
        {"name": "test1"},
        {"name": "test2"}
    ]
    expected = dedent("""\
        [
          {
            name: test1
          },
          {
            name: test2
          },
        ]""").strip()
    assert formatter.format_output(complex_list) == expected


def test_format_dict():
    formatter = ToolOutputFormatter()

    # Simple dict
    assert formatter.format_output({"a": 1}) == "{\n  a: 1\n}"

    # Nested dict
    complex_dict = {
        "name": "test",
        "data": {
            "x": 1,
            "y": [1, 2, 3]
        }
    }
    expected = dedent("""\
        {
          name: test
          data: {
            x: 1
            y: [1, 2, 3]
          }
        }""").strip()
    assert formatter.format_output(complex_dict) == expected


def test_format_file_content():
    formatter = ToolOutputFormatter()

    content = FileContent(
        path="/test/file.py",
        content="def test():\n    pass\n",
        line_range=(1, 2)
    )

    expected = dedent("""\
        File: /test/file.py
        Lines: 1-2
        ----------------------------------------
           1| def test():
           2|     pass""").strip()

    assert formatter.format_output(content) == expected


def test_format_code_block():
    formatter = ToolOutputFormatter()

    code = CodeBlock(
        language="python",
        content="def test():\n    pass"
    )

    expected = dedent("""\
        Language: python
        ----------------------------------------
        def test():
            pass""").strip()

    assert formatter.format_output(code) == expected


def test_format_diff_content():
    formatter = ToolOutputFormatter()

    diff = DiffContent(
        content="@@ -1,1 +1,1 @@\n-old\n+new",
        file_path="test.py",
        stats={"insertions": 1, "deletions": 1}
    )

    expected = dedent("""\
        Diff for: test.py
        Changes: insertions: 1, deletions: 1
        ----------------------------------------
        @@ -1,1 +1,1 @@
        -old
        +new""").strip()

    assert formatter.format_output(diff) == expected


def test_format_styles():
    data = {"name": "test", "values": [1, 2, 3]}

    # Structured (default)
    structured = ToolOutputFormatter(style=FormatStyle.STRUCTURED)
    assert structured.format_output(data) == "{\n  name: test\n  values: [1, 2, 3]\n}"

    # Compact
    compact = ToolOutputFormatter(style=FormatStyle.COMPACT)
    assert compact.format_output(data) == "{'name': 'test', 'values': [1, 2, 3]}"

    # Plain
    plain = ToolOutputFormatter(style=FormatStyle.PLAIN)
    assert plain.format_output(data) == "{\n  name: test\n  values: [1, 2, 3]\n}"

    # Markdown
    code = CodeBlock(language="python", content="print('test')")
    markdown = ToolOutputFormatter(style=FormatStyle.MARKDOWN)
    assert markdown.format_output(code) == "```python\nprint('test')\n```"


def test_content_type_detection():
    formatter = ToolOutputFormatter()

    # File content detection
    file_text = "  1| def test():\n  2|     pass"
    result = formatter.detect_content_type(file_text)
    assert isinstance(result, FileContent)

    # Git diff detection
    diff_text = "diff --git a/test.py b/test.py\n@@ -1,1 +1,1 @@\n-old\n+new"
    result = formatter.detect_content_type(diff_text)
    assert isinstance(result, DiffContent)

    # Code block detection
    code_text = "def test():\n    pass"
    result = formatter.detect_content_type(code_text)
    assert isinstance(result, CodeBlock)

    # Plain text
    text = "Just some text"
    result = formatter.detect_content_type(text)
    assert isinstance(result, str)
