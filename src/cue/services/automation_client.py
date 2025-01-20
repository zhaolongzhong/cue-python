import logging
from typing import Optional
from urllib.parse import urlencode

from .transport import HTTPTransport, ResourceClient, WebSocketTransport
from ..schemas.automation import Automation, AutomationCreate, AutomationUpdate

logger = logging.getLogger(__name__)


class AutomationClient(ResourceClient):
    """Client for automations-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)
        self._default_conversation_id: Optional[str] = None

    def set_default_conversation_id(self, conversation_id):
        self._default_conversation_id = conversation_id

    @property
    def conversation_id(self) -> str:
        return self._default_conversation_id

    async def create(self, automation: AutomationCreate) -> Automation:
        """
        Create a new automation.

        Args:
            automation: AutomationCreate object with automation details

        Returns:
            Automation: Created automation object
        """
        response = await self._http.request("POST", "/automations", data=automation.model_dump())
        if not response:
            logger.error("Create automation failed")
            return None
        return Automation(**response)

    async def get(self, automation_id: str) -> Optional[Automation]:
        """
        Get automation by ID.

        Args:
            automation_id: ID of the automation to retrieve

        Returns:
            Optional[Automation]: Automation if found, None otherwise
        """
        response = await self._http.request("GET", f"/automations/{automation_id}")
        if not response:
            return None
        return Automation(**response)

    async def update(self, automation_id: str, automation: AutomationUpdate) -> Optional[Automation]:
        """
        Update an existing automation.

        Args:
            automation_id: ID of the automation to update
            automation: AutomationUpdate object with fields to update

        Returns:
            Optional[Automation]: Updated automation if successful, None otherwise
        """
        response = await self._http.request("PUT", f"/automations/{automation_id}", data=automation.model_dump())
        if not response:
            return None
        return Automation(**response)

    async def list(
        self,
        *,
        conversation_id: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Automation]:
        """
        List automations with optional filters.

        Args:
            conversation_id: Optional filter by conversation ID
            is_enabled: Optional filter by enabled status
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            List[Automation]: List of automation objects matching the criteria
        """
        # Build query parameters, only including non-None values
        params = {}

        # Always include pagination parameters
        params["skip"] = skip
        params["limit"] = limit

        # Add optional filters only if they're explicitly set
        if conversation_id is not None:
            params["conversation_id"] = conversation_id
        if is_enabled is not None:
            params["is_enabled"] = str(is_enabled).lower()

        # Construct URL with query parameters
        query_string = urlencode(params)
        endpoint = f"/automations?{query_string}"

        response = await self._http.request("GET", endpoint)
        if not response:
            return []
        return [Automation(**asst) for asst in response]

    async def delete(self, automation_id: str) -> bool:
        """
        Delete an automation.

        Args:
            automation_id: ID of the automation to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            await self._http.request("DELETE", f"/automations/{automation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete automation {automation_id}: {str(e)}")
            return False
