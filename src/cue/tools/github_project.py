"""A tool for managing GitHub project items and draft issues.

This tool provides a complete interface for managing GitHub project items, including
draft issues and project cards. It supports listing, creating, updating, searching, and
getting items with automatic caching for efficient operation.

Key Features:
- List all project items with IDs and titles
- Create new draft issues and project items
- Update existing items (title and body)
- Search items by title
- Get detailed item information
- Built-in caching for performance
- Proper error handling and validation
"""

from typing import Dict, Literal, ClassVar, Optional, get_args

from .base import BaseTool, ToolError, ToolResult
from .github.project import ProjectItem, GitHubProject

Command = Literal["list", "create", "update", "search", "get"]
Status = Literal["Backlog", "Todo", "In Progress", "Done"]
VALID_STATUS_VALUES = list(get_args(Status))


class GitHubProjectTool(BaseTool):
    """
    A tool for managing GitHub project items and draft issues.
    Supports listing, creating, updating, searching and getting project items.
    """

    name: ClassVar[Literal["github_project"]] = "github_project"

    def __init__(self):
        """Initialize tool with default project settings."""
        self._function = self.github_project
        self._projects: Dict[int, GitHubProject] = {}
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        project_number: int,
        item_id: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the tool with the given arguments."""
        return await self.github_project(
            command=command,
            project_number=project_number,
            item_id=item_id,
            title=title,
            body=body,
            query=query,
            **kwargs,
        )

    def validate_status(self, status: Optional[str]) -> None:
        """Validate status value against allowed options."""
        if status is not None and status not in VALID_STATUS_VALUES:
            valid_options = '", "'.join(VALID_STATUS_VALUES)
            raise ToolError(f'Invalid status value: "{status}". Valid options are: "{valid_options}"')

    def _get_project(self, project_number: int) -> GitHubProject:
        """Get or create GitHubProject instance."""
        if project_number not in self._projects:
            self._projects[project_number] = GitHubProject(project_number)
        return self._projects[project_number]

    def _format_items(self, items: list[ProjectItem], project_number: int, full_body: bool = False) -> str:
        """Format items list for display.

        Args:
            items: List of ProjectItems to format
            project_number: Project number for context
            full_body: If True, show complete body text (for single item display)
        """
        if not items:
            return f"No items found in project {project_number}"

        result = []
        for item in items:
            result.append(f"Item Details in Project {project_number}:")
            result.append("-" * 40)  # Separator
            result.append(f"Title    : {item.title}")
            result.append(f"Item ID  : {item.item_id}")
            result.append(f"Content ID: {item.content_id}")
            result.append(f"Type     : {item.type}")
            if item.status:
                result.append(f"Status   : {item.status}")
            if item.body:
                # Show full body for single item view, truncate for lists
                body_text = item.body if (full_body or len(items) == 1) else f"{item.body[:100]}..."
                result.append(f"Body     : {body_text}")
            result.append("-" * 40)  # Separator
            result.append("")  # Empty line between items
        return "\n".join(result)

    async def github_project(
        self,
        *,
        command: Command,
        project_number: int,
        item_id: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        status: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Perform operations on GitHub project items.

        This method serves as the main entry point for all GitHub project operations,
        delegating to specific methods based on the command provided.

        Args:
            command (Command): The operation to perform. Must be one of:
                - "list": List all items in the project
                - "create": Create a new project item
                - "update": Update an existing item (requires PVTI_ ID)
                - "search": Search for items by title
                - "get": Get details of a specific item (requires PVTI_ ID)
            project_number (int): The GitHub project number to operate on
            item_id (Optional[str]): For update/get commands, must be the PVTI_ ID (not DI_ ID).
                PVTI_ IDs are shown in the list/search results and are required for item operations.
                Example: "PVTI_lAHOADKj2c4Ag2eizgVXKCg"
            title (Optional[str]): Item title. Required for create, optional for update.
            body (Optional[str]): Item body content. Optional for create and update.
            status (Optional[str]): Item status. Optional for update.
            query (Optional[str]): Search query string. Required for search command.
            **kwargs: Additional keyword arguments (unused)

        Returns:
            ToolResult: Operation result containing either:
                - Success message for create/update operations
                - Item details for get operation
                - List of items for list/search operations

        Note:
            There are two types of IDs in the GitHub Projects system:
            1. PVTI_ IDs: Used for project item operations (get/update)
            2. DI_ IDs: Internal content IDs, not used for item operations
            Always use the PVTI_ ID shown in list/search results for get/update commands.

        Raises:
            ToolError: If:
                - Required parameters are missing for specific commands
                - Invalid item_id format (must start with PVTI_)
                - Project number is invalid
                - Item not found
                - API operation fails
        """
        # Validate project number
        if project_number <= 0:
            raise ToolError("Project number must be a positive integer")

        # Validate item_id format for commands that require it
        if command in ("get", "update") and item_id:
            if not item_id.startswith("PVTI_"):
                raise ToolError(
                    "Invalid item_id format. Must use PVTI_ ID (not DI_ ID) for get/update operations. "
                    "Use the list command to see available PVTI_ IDs."
                )

        try:
            project = self._get_project(project_number)

            if command == "list":
                items = project.list_items()
                return ToolResult(output=self._format_items(items, project_number))

            elif command == "create":
                if not title:
                    raise ToolError("Parameter `title` is required for command: create")

                item = project.create_item(title=title, body=body)
                return ToolResult(
                    output=f"Created item in project {project_number}:\n{self._format_items([item], project_number)}"
                )

            elif command == "update":
                if not item_id:
                    raise ToolError("Parameter `item_id` is required for command: update")
                if not title and not body:
                    raise ToolError("Either `title` or `body` must be provided for command: update")

                try:
                    # Validate status value early if provided
                    self.validate_status(status)
                    # First get the current item to preserve title if not updating it
                    current_item = project.get_item(item_id)
                    if not current_item:
                        raise ToolError(f"Item {item_id} not found in project {project_number}")

                    # If no title provided, use existing title
                    update_title = title if title is not None else current_item.title
                    update_body = body if body is not None else current_item.body
                    update_status = status if status is not None else current_item.status

                    item = project.update_item(
                        item_id=item_id,
                        title=update_title,  # Always provide title
                        body=update_body,
                        status=update_status,
                    )
                    return ToolResult(
                        output=f"Updated item in project {project_number}:\n{self._format_items([item], project_number)}"
                    )
                except Exception as e:
                    # Raise ToolError with the original error message
                    raise ToolError(f"Failed to update item {item_id}: {str(e)}")

            elif command == "search":
                if not query:
                    raise ToolError("Parameter `query` is required for command: search")

                items = project.search_items(query)
                return ToolResult(
                    output=f"Found {len(items)} items in project {project_number}:\n{self._format_items(items, project_number)}"
                )

            elif command == "get":
                if not item_id:
                    raise ToolError("Parameter `item_id` is required for command: get")

                item = project.get_item(item_id)
                if not item:
                    raise ToolError(f"Item {item_id} not found in project {project_number}")
                return ToolResult(output=self._format_items([item], project_number))

            raise ToolError(
                f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: {', '.join(get_args(Command))}"
            )

        except Exception as e:
            raise ToolError(f"Failed to execute {command} on project {project_number}: {str(e)}")
