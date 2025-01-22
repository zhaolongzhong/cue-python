import json
import logging
from typing import Union, Literal, ClassVar, Optional, get_args
from pathlib import Path

from .base import BaseTool, ToolError, ToolResult
from ..schemas import AssistantMemoryCreate, AssistantMemoryUpdate
from ..services import MemoryClient
from ..tools.memory_formatter import MemoryFormatter
from ..tools.memory_validator import ParameterValidator

logger = logging.getLogger(__name__)

Command = Literal[
    "create",
    "recall",
    "update",
    "delete",
    "view",
]


class MemoryTool(BaseTool):
    """
    An memory tool that allows the agent to store, recall, and update memories.
    """

    name: ClassVar[Literal["memory"]] = "memory"

    _file_history: dict[Path, list[str]]

    def __init__(self, memory_service: Optional[MemoryClient]):
        self._function = self.memory
        self.memory_client = memory_service
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        new_str: Optional[str] = None,
        memory_id: Optional[str] = None,
        assistant_id: str = None,
        **kwargs,
    ):
        return await self.memory(
            assistant_id=assistant_id,
            command=command,
            query=query,
            limit=limit,
            new_str=new_str,
            memory_id=memory_id,
            **kwargs,
        )

    async def memory(
        self,
        *,
        assistant_id: str = None,
        command: Command,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        memory_id: Optional[Union[str, list[str]]] = None,
        new_str: Optional[str] = None,
        **kwargs,
    ):
        """Perform memory operations like "view", "create", "recall", "update", and "delete" memory."""
        if self.memory_client is None:
            error_msg = "Memory tool is called but external memory is not enabled."
            logger.error(error_msg)
            raise ToolError(error_msg)
        if assistant_id is None:
            raise ToolError("Assistant ID is required for memory operations.")
        if command == "view":
            limit = ParameterValidator.safe_int(limit, default=10)
            return await self.view(assistant_id=assistant_id, memory_id=memory_id, limit=limit.value)
        elif command == "recall":
            limit = ParameterValidator.safe_int(limit, default=10)
            return await self.recall(assistant_id=assistant_id, query=query, limit=limit.value)
        elif command == "create":
            if not new_str:
                raise ToolError("Parameter `new_str` is required for command: create")
            return await self.create(assistant_id=assistant_id, new_str=new_str)
        elif command == "update":
            if not new_str:
                raise ToolError("Parameter `new_str` is required for command: update")
            if not memory_id:
                raise ToolError("Parameter `memory_id` is required for command: update")
            return await self.update(assistant_id=assistant_id, new_str=new_str, memory_id=memory_id)
        elif command == "delete":
            return await self.delete(assistant_id=assistant_id, memory_id=memory_id)
        raise ToolError(
            f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: "
            f"{', '.join(get_args(Command))}"
        )

    async def view(
        self, assistant_id: str = None, memory_id: Optional[str] = None, limit: Optional[int] = 10
    ) -> ToolResult:
        response = []
        memory_ids = []
        if memory_id:
            memory_ids = ParameterValidator.safe_string_list(memory_id, ",")

        if memory_ids:
            for memory_id in memory_ids:
                memory = await self.memory_client.get_memory(assistant_id=assistant_id, memory_id=memory_id)
                response.append(memory)
        else:
            response = await self.memory_client.get_memories(assistant_id=assistant_id, limit=limit)
        response.reverse()
        memory_contents = MemoryFormatter.format_memory_list(
            memories=response,
            include_scores=False,
            include_metadata=False,
            date_format="%Y-%m-%d %H:%M:%S",
            separator="\n---\n",
            include_summary=False,
        )
        logger.debug(f"view memories: {len(response)}, memory_contents: {memory_contents[:500]}")
        return ToolResult(output=memory_contents)

    async def get_recent_memories(self, assistant_id: str, limit: Optional[int] = 10) -> dict[str, str]:
        """Return a dict formatted memories entry in asc order by updated_at"""
        if not self.memory_client:
            return
        response = await self.memory_client.get_memories(assistant_id=assistant_id, limit=limit)
        memory_dict = {
            memory.id: MemoryFormatter.format_single_memory(
                memory,
                include_score=False,
                include_metadata=False,
                date_format="%Y-%m-%d %H:%M:%S",
                indent_level=2,
            )
            for memory in reversed(response)
        }
        return memory_dict

    async def recall(self, assistant_id: str, query: str, limit: Optional[int] = 10) -> ToolResult:
        """Implement the recall command"""
        response = await self.memory_client.search_memories(assistant_id=assistant_id, query=query, limit=limit)
        retrieved_memories = response.memories

        if len(retrieved_memories) == 0:
            return ToolResult(output=f"No memories found for query: {query}, or use a different query.")

        memory_contents = MemoryFormatter.format_memory_list(
            memories=retrieved_memories,
            include_scores=True,
            include_metadata=False,
            date_format="%Y-%m-%d %H:%M:%S",
            separator="\n---\n",
            include_summary=False,
        )
        logger.debug(f"recall memories size: {len(retrieved_memories)}, memory_contents: {memory_contents[:500]}...")
        return ToolResult(output=memory_contents)

    async def update(
        self,
        assistant_id: str,
        memory_id: str,
        new_str: Optional[str],
    ):
        """Implement the update command, which replaces memory content with new_str in the given memory"""
        try:
            await self.memory_client.update_memory(
                assistant_id=assistant_id, memory_id=memory_id, memory=AssistantMemoryUpdate(content=new_str)
            )
            success_msg = f"The memory<id: {memory_id}> has been edited. "
            return ToolResult(output=success_msg)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to create a memory for {new_str}") from None

    async def create(self, assistant_id: str, new_str: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            entry = await self.memory_client.create(
                assistant_id=assistant_id, memory=AssistantMemoryCreate(content=new_str)
            )
            return ToolResult(output=f"Memory created successfully with id: {entry.id}")
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to create a memory for {new_str}") from None

    async def delete(self, assistant_id: str, memory_id: str) -> ToolResult:
        """Delete one or more memories. Can accept either a single memory ID or a list of memory IDs."""
        try:
            result = ParameterValidator.safe_string_list(memory_id, ",")
            if not result.is_valid:
                raise ToolError("No memory IDs provided for deletion")

            memory_ids = result.value
            result = await self.memory_client.delete_memories(assistant_id=assistant_id, memory_ids=memory_ids)

            # Format the response message based on the result
            if result["success"]:
                if len(memory_ids) == 1:
                    message = f"Successfully deleted memory {memory_ids[0]}"
                else:
                    message = f"Successfully deleted {result['deleted_count']} memories"
            else:
                failed_count = len(result.get("failed_deletions", []))
                success_count = result.get("deleted_count", 0)

                if success_count > 0:
                    message = f"Partially successful: {success_count} memories deleted, {failed_count} failed"
                else:
                    message = "Failed to delete memories"

                # Add failure details if any
                if result.get("failed_deletions"):
                    failure_details = "\nFailure details:\n" + "\n".join(
                        f"- Memory {failure['id']}: {failure['reason']}" for failure in result["failed_deletions"]
                    )
                    message += failure_details

            result = {
                "success": result["success"],
                "deleted_count": result["deleted_count"],
                "successful_deletions": result.get("successful_deletions", []),
                "failed_deletions": result.get("failed_deletions", []),
            }
            logger.debug(f"Delete memory result: {json.dumps(result, indent=4)}")
            return ToolResult(
                output=message,
            )

        except Exception as e:
            # Format the error message based on whether it was a single or bulk deletion
            id_str = memory_id if isinstance(memory_id, str) else ", ".join(memory_id)
            raise ToolError(
                f"Failed to delete memory{'ies' if isinstance(memory_id, list) else ''} ({id_str}): {str(e)}"
            )
