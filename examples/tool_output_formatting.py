"""
Example usage of tool output formatting
"""
from cue.schemas.tool_output_formatter import CodeBlock, DiffContent, FileContent, FormatStyle, ToolOutputFormatter


def main():
    # Create formatter instances for different styles
    structured = ToolOutputFormatter(style=FormatStyle.STRUCTURED)
    compact = ToolOutputFormatter(style=FormatStyle.COMPACT)
    markdown = ToolOutputFormatter(style=FormatStyle.MARKDOWN)

    # Example file content
    file_content = FileContent(
        path="/example/test.py",
        content="\n".join([
            "def example():",
            "    print('Hello World')",
            "    return True"
        ]),
        line_range=(1, 3)
    )

    print("File Content (Structured):")
    print(structured.format_output(file_content))
    print("\nFile Content (Compact):")
    print(compact.format_output(file_content))

    # Example code block
    code = CodeBlock(
        language="python",
        content="\n".join([
            "def greet(name: str) -> str:",
            "    return f'Hello {name}!'"
        ])
    )

    print("\nCode Block (Markdown):")
    print(markdown.format_output(code))
    print("\nCode Block (Structured):")
    print(structured.format_output(code))

    # Example diff
    diff = DiffContent(
        content="\n".join([
            "@@ -1,3 +1,3 @@",
            " def example():",
            "-    print('Hello')",
            "+    print('Hello World')",
            "     return True"
        ]),
        file_path="test.py",
        stats={"insertions": 1, "deletions": 1}
    )

    print("\nDiff Content:")
    print(structured.format_output(diff))

    # Example complex data
    data = {
        "files": [
            {"name": "test.py", "lines": 100},
            {"name": "main.py", "lines": 50}
        ],
        "stats": {
            "total_lines": 150,
            "languages": ["Python"]
        }
    }

    print("\nComplex Data (Structured):")
    print(structured.format_output(data))
    print("\nComplex Data (Compact):")
    print(compact.format_output(data))

if __name__ == "__main__":
    main()
