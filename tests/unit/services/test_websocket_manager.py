"""Tests for WebSocket manager singleton implementation."""

import asyncio
from typing import Any, Dict, Callable, Awaitable

import pytest

from cue.services.transport import WebSocketTransport
from cue.services.websocket_manager import WebSocketManager
from cue.services.transport.websocket_connection_error import WebSocketConnectionError


class MockWebSocketTransport(WebSocketTransport):
    """Mock WebSocket transport for testing."""

    def __init__(self, fail_connect: bool = False):
        self.messages = []
        self.connected = False
        self.fail_connect = fail_connect
        self.close_called = False

    async def connect(self) -> None:
        if self.fail_connect:
            raise WebSocketConnectionError("Connection failed")
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self.close_called = True

    async def send(self, message: str) -> None:
        if not self.connected:
            raise WebSocketConnectionError("Not connected")
        self.messages.append(message)

    async def receive(self) -> Dict[str, Any]:
        if not self.connected:
            raise WebSocketConnectionError("Not connected")
        return {"type": "test_message", "data": "test"}

    async def ping(self) -> None:
        if not self.connected:
            raise WebSocketConnectionError("Not connected")


@pytest.fixture
def message_handlers() -> Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]]:
    """Provide message handlers fixture."""
    async def test_handler(message: Dict[str, Any]) -> None:
        pass

    return {
        "test_message": test_handler
    }


def test_websocket_manager_singleton(message_handlers):
    """Test WebSocket manager singleton pattern."""
    # Initially no instance should exist
    assert WebSocketManager.get_instance() is None

    transport1 = MockWebSocketTransport()
    transport2 = MockWebSocketTransport()

    # Create first instance
    manager1 = WebSocketManager(
        ws_transport=transport1,
        message_handlers=message_handlers,
        auto_reconnect=True
    )

    # get_instance should now return our manager
    assert WebSocketManager.get_instance() is manager1

    # Create second instance with different parameters
    manager2 = WebSocketManager(
        ws_transport=transport2,
        message_handlers=None,
        auto_reconnect=False
    )

    # Verify all references point to same instance
    assert manager1 is manager2
    assert WebSocketManager.get_instance() is manager1
    assert WebSocketManager.get_instance() is manager2

    # Verify first initialization parameters are kept
    assert manager1._transport is transport1
    assert manager1._message_handlers == message_handlers
    assert manager1._auto_reconnect is True

    # Second initialization should not change parameters
    assert manager2._transport is not transport2
    assert manager2._transport is transport1
    assert manager2._message_handlers == message_handlers
    assert manager2._auto_reconnect is True

    # Test getting instance after initialization
    manager3 = WebSocketManager.get_instance()
    assert manager3 is manager1


@pytest.mark.asyncio
async def test_websocket_manager_metrics(message_handlers):
    """Test WebSocket manager metrics tracking."""
    transport = MockWebSocketTransport()
    manager = WebSocketManager(
        ws_transport=transport,
        message_handlers=message_handlers
    )

    # Initial metrics state
    assert manager.metrics.connection_attempts == 0
    assert manager.metrics.successful_messages_sent == 0
    assert manager.metrics.failed_messages == 0
    assert manager.metrics.last_connected_at is None
    assert manager.metrics.last_disconnected_at is None
    assert manager.metrics.last_error is None

    # Connect and verify metrics
    await manager.connect()
    assert manager.metrics.connection_attempts == 1
    assert manager.metrics.last_connected_at is not None
    assert manager.metrics.last_error is None

    # Send a message and verify metrics
    await manager.send_message("test")
    assert manager.metrics.successful_messages_sent == 1
    assert manager.metrics.failed_messages == 0

    # Disconnect and verify metrics
    await manager.disconnect()
    assert manager.metrics.last_disconnected_at is not None


@pytest.mark.asyncio
async def test_websocket_manager_error_metrics(message_handlers):
    """Test WebSocket manager error metrics tracking."""
    transport = MockWebSocketTransport(fail_connect=True)
    manager = WebSocketManager(
        ws_transport=transport,
        message_handlers=message_handlers,
        auto_reconnect=True,
        max_reconnect_attempts=2
    )

    # Test connection failure
    with pytest.raises(Exception):
        await manager.connect()
    assert manager.metrics.connection_attempts >= 1
    assert manager.metrics.last_error is not None
    assert "Connection failed" in manager.metrics.last_error


@pytest.mark.asyncio
async def test_message_queue_overflow(message_handlers):
    """Test message queue overflow handling."""
    transport = MockWebSocketTransport()
    manager = WebSocketManager(
        ws_transport=transport,
        message_handlers=message_handlers,
        message_queue_size=2  # Small queue for testing
    )

    # Fill queue
    await manager.send_message("msg1")
    await manager.send_message("msg2")

    # Next message should raise queue full error
    with pytest.raises(RuntimeError, match="Message queue full"):
        await manager.send_message("msg3")

    assert manager.metrics.failed_messages == 1


@pytest.mark.asyncio
async def test_singleton_concurrent_access():
    """Test concurrent access to WebSocket manager singleton."""
    transport = MockWebSocketTransport()

    async def create_manager(i: int):
        """Create manager instance with delay."""
        await asyncio.sleep(0.1 * i)  # Stagger creation
        return WebSocketManager(ws_transport=transport)

    # Create managers concurrently
    managers = await asyncio.gather(*[
        create_manager(i) for i in range(5)
    ])

    # Verify all instances are the same
    first = managers[0]
    for manager in managers[1:]:
        assert manager is first
