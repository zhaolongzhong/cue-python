from typing import Union
from textwrap import dedent, indent

from ..schemas.assistant_memory import RelevantMemory, AssistantMemory


class MemoryFormatter:
    @staticmethod
    def format_single_memory(
        memory: Union[AssistantMemory, RelevantMemory],
        include_score: bool = True,
        include_metadata: bool = False,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        indent_level: int = 0,
    ) -> str:
        """
        Format a single memory into a readable string

        Args:
            memory: RelevantMemory object to format
            include_score: Whether to include similarity score
            include_metadata: Whether to include metadata
            date_format: Format string for datetime
            indent_level: Number of spaces to indent

        Returns:
            Formatted string representation of the memory
        """
        # Base memory information
        memory_parts = [
            f"Content: {memory.content}",
            f"ID: {memory.id}",
            f"Date: {memory.updated_at.strftime(date_format)}",
        ]

        # Add optional score if present and requested
        if include_score and memory.similarity_score is not None:
            memory_parts.append(f"Relevance: {memory.similarity_score:.2%}")

        # Add metadata if present and requested
        if include_metadata and memory.metadata:
            metadata_str = "\n".join(f"  {key}: {value}" for key, value in memory.metadata.items())
            memory_parts.append(f"Metadata:\n{metadata_str}")

        # Join parts and apply indentation
        memory_str = "\n".join(memory_parts)
        if indent_level > 0:
            memory_str = indent(memory_str, " " * indent_level)

        return memory_str

    @staticmethod
    def format_memory_list(
        memories: list[Union[AssistantMemory, RelevantMemory]],
        include_scores: bool = True,
        include_metadata: bool = False,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        separator: str = "\n---\n",
        include_summary: bool = True,
    ) -> str:
        """
        Format a list of memories into a readable string

        Args:
            memories: List of RelevantMemory objects
            include_scores: Whether to include similarity scores
            include_metadata: Whether to include metadata
            date_format: Format string for datetime
            separator: String to use between memories
            include_summary: Whether to include summary header

        Returns:
            Formatted string representation of all memories
        """
        if not memories:
            return "No memories found."

        # Create summary if requested
        parts = []
        if include_summary:
            summary = dedent(f"""
                Memory Search Results
                Total Memories: {len(memories)}
                Date Range: {memories[-1].updated_at.strftime(date_format)} to
                {memories[0].updated_at.strftime(date_format)}
                """).strip()
            parts.append(summary)

        # Format each memory
        memory_strings = [
            MemoryFormatter.format_single_memory(
                memory,
                include_score=include_scores,
                include_metadata=include_metadata,
                date_format=date_format,
                indent_level=2,
            )
            for memory in memories
        ]

        # Join everything together
        parts.extend(memory_strings)
        return separator.join(parts)
