"""Base client for resource operations."""
from typing import Optional

from .transport import HTTPTransport, WebSocketTransport


class ResourceClient:
    """Base client for resource operations."""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        """Initialize resource client."""
        self._http = http
        self._ws = ws
