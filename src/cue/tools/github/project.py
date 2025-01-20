"""GitHub Project management tool."""

import json
import logging
import subprocess
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProjectItem:
    """Represents an item in a GitHub project."""

    item_id: str  # PVTI_ id
    content_id: str  # DI_ id for draft issues
    title: str
    body: Optional[str] = None
    status: Optional[str] = None
    type: str = "DraftIssue"

    def __str__(self) -> str:
        """Return string representation of the item."""
        return f"{self.title} ({self.item_id})"

    def to_dict(self) -> dict:
        """Convert item to dictionary."""
        return {
            "item_id": self.item_id,
            "content_id": self.content_id,
            "title": self.title,
            "body": self.body,
            "status": self.status,
            "type": self.type,
        }


class GitHubProject:
    """Manages GitHub project operations."""

    def __init__(self, project_number: int, owner: str = "@me"):
        """Initialize GitHub project tool.

        Args:
            project_number: The project number
            owner: Project owner, defaults to "@me" for current user
        """
        self.project_number = project_number
        self.owner = owner
        self._items_cache: Optional[dict[str, ProjectItem]] = None
        self._fields_cache: Optional[dict[str, str]] = None  # name -> id mapping

        # Get project ID during initialization
        cmd = ["project", "view", str(project_number), "--owner", owner, "--format", "json"]
        data = self._run_gh_cmd(cmd)
        self.project_id = data.get("id")
        if not self.project_id:
            raise ValueError(f"Could not find project with number {project_number}")

        # Initialize fields cache
        self.refresh_fields_cache()

    def _run_gh_cmd(self, cmd: list[str]) -> dict:
        """Run GitHub CLI command and return JSON response.

        Args:
            cmd: Command parts to pass to gh CLI

        Returns:
            JSON response from gh CLI

        Raises:
            Exception: If gh command fails
        """
        try:
            logger.debug(f"Running gh command: gh {' '.join(cmd)}")
            result = subprocess.run(["gh"] + cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout) if result.stdout else {}
        except subprocess.CalledProcessError as e:
            error_msg = f"GitHub CLI error: {e.stderr}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse GitHub CLI response: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def refresh_fields_cache(self) -> None:
        """Refresh the cache of project fields."""
        cmd = ["project", "field-list", str(self.project_number), "--owner", self.owner, "--format", "json"]
        data = self._run_gh_cmd(cmd)

        self._fields_cache = {}
        for field in data.get("fields", []):
            self._fields_cache[field["name"]] = field["id"]

        logger.debug(f"Cached {len(self._fields_cache)} project fields")

    def refresh_cache(self) -> None:
        """Refresh the local cache of project items."""
        cmd = ["project", "item-list", str(self.project_number), "--owner", self.owner, "--format", "json"]
        data = self._run_gh_cmd(cmd)

        self._items_cache = {}
        for item in data.get("items", []):
            content = item.get("content", {})
            project_item = ProjectItem(
                item_id=item["id"],
                content_id=content.get("id", ""),
                title=item.get("title", content.get("title", "")),
                body=content.get("body", ""),
                status=item.get("status"),
                type=content.get("type", "DraftIssue"),
            )
            self._items_cache[item["id"]] = project_item

        logger.debug(f"Cached {len(self._items_cache)} project items")

    def get_item(self, item_id: str) -> Optional[ProjectItem]:
        """Get a project item by its PVTI_ ID.

        Args:
            item_id: The PVTI_ id of the item

        Returns:
            ProjectItem if found, None otherwise
        """
        if not self._items_cache:
            self.refresh_cache()
        return self._items_cache.get(item_id)

    def update_item(
        self, item_id: str, title: Optional[str] = None, body: Optional[str] = None, status: Optional[str] = None
    ) -> ProjectItem:
        """Update a project item.

        Args:
            item_id: The PVTI_ id of the item
            title: New title (optional)
            body: New body content (optional)
            status: New status (optional)

        Returns:
            Updated ProjectItem

        Raises:
            ValueError: If item not found
        """
        item = self.get_item(item_id)
        if not item:
            raise ValueError(f"Item {item_id} not found")

        # Always include current title if not updating
        update_title = title if title is not None else item.title

        # For draft issue title updates
        if title is not None:
            if not item.content_id or not item.content_id.startswith("DI_"):
                raise ValueError(f"Item {item_id} is not a draft issue")

            # Use correct pattern: gh project item-edit --id <DI_ID> --title "<NEW_TITLE>" --format json
            cmd = ["project", "item-edit", "--id", item.content_id, "--title", title, "--format", "json"]
            self._run_gh_cmd(cmd)
            logger.info(f"Updated title for item {item_id}")

        # For body updates
        if body is not None:
            if not item.content_id or not item.content_id.startswith("DI_"):
                raise ValueError(f"Item {item_id} is not a draft issue")

            # Use correct pattern: gh project item-edit --id <DI_ID> --title "<CURRENT_TITLE>" --body "<NEW_BODY>"
            # --format json
            cmd = [
                "project",
                "item-edit",
                "--id",
                item.content_id,
                "--title",
                update_title,
                "--body",
                body,
                "--format",
                "json",
            ]
            self._run_gh_cmd(cmd)
            logger.info(f"Updated body for item {item_id}")

        # Update status if provided
        if status:
            if not self._fields_cache:
                self.refresh_fields_cache()

            # Get status field details
            cmd = ["project", "field-list", str(self.project_number), "--owner", self.owner, "--format", "json"]
            data = self._run_gh_cmd(cmd)

            # Find status field and its options
            status_field = None
            for field in data.get("fields", []):
                if field["name"] == "Status":
                    status_field = field
                    break

            if not status_field:
                raise ValueError("Status field not found in project")

            # Find matching status option
            status_option = None
            for option in status_field.get("options", []):
                if option["name"].lower() == status.lower():
                    status_option = option
                    break

            if not status_option:
                valid_options = [opt["name"] for opt in status_field.get("options", [])]
                raise ValueError(f"Invalid status value. Valid options are: {', '.join(valid_options)}")

            # Update status using field ID and option ID
            cmd = [
                "project",
                "item-edit",
                "--id",
                item_id,
                "--field-id",
                status_field["id"],
                "--project-id",
                self.project_id,
                "--single-select-option-id",
                status_option["id"],
                "--format",
                "json",
            ]
            self._run_gh_cmd(cmd)
            logger.info(f"Updated status for item {item_id} to {status}")

        # Refresh cache and return updated item
        self.refresh_cache()
        updated_item = self.get_item(item_id)
        if not updated_item:
            raise Exception(f"Failed to retrieve updated item {item_id}")
        return updated_item

    def create_item(self, title: str, body: Optional[str] = None) -> ProjectItem:
        """Create a new draft issue in the project.

        Args:
            title: Item title
            body: Item body content (optional)

        Returns:
            ProjectItem: The created item

        Raises:
            Exception: If creation fails or item not found after creation
        """
        cmd = ["project", "item-create", str(self.project_number), "--owner", self.owner, "--title", title]
        if body:
            cmd.extend(["--body", body])

        self._run_gh_cmd(cmd)
        logger.info(f"Created new project item: {title}")

        self.refresh_cache()

        # Find and return the newly created item
        for item in self._items_cache.values():
            if item.title == title:
                return item
        raise Exception("Failed to find created item")

    def list_items(self) -> list[ProjectItem]:
        """List all items in the project.

        Returns:
            List of all ProjectItems
        """
        self.refresh_cache()
        return list(self._items_cache.values())

    def search_items(self, query: str) -> list[ProjectItem]:
        """Search for items by title (case-insensitive).

        Args:
            query: Search string

        Returns:
            List of matching ProjectItems
        """
        if not self._items_cache:
            self.refresh_cache()

        query = query.lower()
        return [item for item in self._items_cache.values() if query in item.title.lower()]
