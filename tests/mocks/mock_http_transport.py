"""Mock HTTP transport for testing."""
from typing import Any, Dict, Optional
from datetime import datetime

from cue.services.transport import HTTPTransport

from .mock_task_server import MockTaskServer


class MockHTTPTransport(HTTPTransport):
    """Mock HTTP transport implementation."""

    def __init__(self):
        """Initialize mock transport."""
        self.task_server = MockTaskServer()
        self.is_server_available = True

    async def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Process a mock request."""
        parts = endpoint.strip('/').split('/')
        resource = parts[0]

        if resource != 'tasks':
            raise ValueError(f"Unknown resource: {resource}")

        # Extract task_id if present
        task_id = parts[1] if len(parts) > 1 else None

        if method == "GET":
            if task_id:
                task = await self.task_server.get_task(task_id)
                return task.model_dump() if task else None
            elif endpoint.endswith('/due'):
                before = params.get('before')
                if before:
                    before = datetime.fromisoformat(before)
                tasks = await self.task_server.get_due_tasks(before)
                return [t.model_dump() for t in tasks]
            else:
                skip = params.get('skip', 0)
                limit = params.get('limit', 100)
                tasks = await self.task_server.list_tasks(skip, limit)
                return [t.model_dump() for t in tasks]

        elif method == "POST":
            from cue.schemas.scheduled_task import ScheduledTaskCreate
            task = ScheduledTaskCreate(**data)
            result = await self.task_server.create_task(task)
            return result.model_dump()

        elif method == "PUT":
            from cue.schemas.scheduled_task import ScheduledTaskUpdate
            task = ScheduledTaskUpdate(**data)
            result = await self.task_server.update_task(task_id, task)
            return result.model_dump() if result else None

        elif method == "DELETE":
            await self.task_server.delete_task(task_id)
            return None

        raise ValueError(f"Unknown method: {method}")
