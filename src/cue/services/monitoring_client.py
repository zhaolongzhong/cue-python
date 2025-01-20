import json
import logging
from typing import Any, Optional
from datetime import datetime

from pydantic import BaseModel

from cue.schemas.monitoring import ErrorType

from ..schemas import ErrorReport, ErrorReportResponse
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class MonitoringContext(BaseModel):
    """Holds monitoring context information"""

    assistant_id: Optional[str] = None
    conversation_id: Optional[str] = None
    component_name: Optional[str] = None


class MonitoringClient(ResourceClient):
    """Client for message-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)
        self.default_assistant_id: Optional[str] = None
        self.default_conversation_id: Optional[str] = None
        self.context = MonitoringContext()

    def set_context(
        self,
        assistant_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        component_name: Optional[str] = None,
    ):
        """Set default context for monitoring"""
        self.context = MonitoringContext(
            assistant_id=assistant_id,
            conversation_id=conversation_id,
            component_name=component_name or self.context.component_name,
        )

    async def _report_error(self, error: ErrorReport) -> ErrorReportResponse:
        json_data = json.loads(error.model_dump_json())
        response = await self._http.request("POST", "/monitoring/errors", data=json_data)
        if not error.conversation_id:
            error.conversation_id = self.default_conversation_id
        return ErrorReportResponse(**response)

    async def report_error(
        self,
        message: str,
        error_type: ErrorType,
        severity: str = "error",
        metadata: Optional[dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
    ) -> ErrorReportResponse:
        """
        Report an error with current context

        Args:
            message: Error message
            error_type: Type of error (system, agent, tool, llm, transfer)
            severity: Error severity (info, warning, error, critical)
            metadata: Additional error context
            conversation_id: Override default conversation_id
            assistant_id: Override default assistant_id
        """
        try:
            error = ErrorReport(
                type=error_type,
                message=message,
                severity=severity,
                conversation_id=conversation_id or self.context.conversation_id,
                agent_id=assistant_id or self.context.assistant_id,
                timestamp=datetime.now(),
                metadata={"component": self.context.component_name, **(metadata or {})},
            )

            return await self._report_error(error=error)

        except Exception as e:
            logger.error(f"Failed to report error: {str(e)}", exc_info=True)
            # Return a basic response to avoid breaking the caller
            return ErrorReportResponse(status="failed", timestamp=datetime.now().isoformat())

    async def report_exception(
        self,
        exc: Exception,
        error_type: ErrorType = ErrorType.SYSTEM,
        severity: str = "error",
        additional_context: Optional[dict[str, Any]] = None,
    ) -> ErrorReportResponse:
        """
        Report an exception with automatic context capture
        """
        metadata = {"error_type": exc.__class__.__name__, "error_trace": str(exc), **(additional_context or {})}

        return await self.report_error(message=str(exc), error_type=error_type, severity=severity, metadata=metadata)
