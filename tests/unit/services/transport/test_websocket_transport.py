import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import WSMessage, WSMsgType, ClientWebSocketResponse

from src.cue.services.transport.websocket_transport import AioHTTPWebSocketTransport


@pytest.fixture
def transport():
    """Create a WebSocket transport instance for testing."""
    return AioHTTPWebSocketTransport(
        ws_url="ws://test.com",
        client_id="test_client",
        api_key="test_key",
        max_retries=2,
        retry_delay=0.1  # Short delay for faster tests
    )

@pytest.mark.asyncio
async def test_listen_reconnects_on_closing_transport_error(transport):
    """Test that _listen() attempts to reconnect when encountering 'Cannot write to closing transport' error."""
    # Mock the WebSocket connection
    transport.ws = AsyncMock(spec=ClientWebSocketResponse)
    transport._connected = True

    # Set up the ws.__aiter__ to raise the error we want to test
    async def mock_aiter():
        raise Exception("Cannot write to closing transport")
    transport.ws.__aiter__ = mock_aiter

    # Mock connect() to track calls and reset error state
    transport.connect = AsyncMock()
    transport._handle_disconnect = AsyncMock()

    # Start _listen() in a task we can cancel
    listen_task = asyncio.create_task(transport._listen())

    # Wait a short time for the error to be handled
    await asyncio.sleep(0.2)

    # Cancel the infinite loop
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass

    # Verify reconnection was attempted
    assert transport.connect.called
    assert transport._handle_disconnect.called

@pytest.mark.asyncio
async def test_listen_handles_close_message(transport):
    """Test that _listen() properly handles WebSocket CLOSE messages."""
    # Mock the WebSocket connection
    transport.ws = AsyncMock(spec=ClientWebSocketResponse)
    transport._connected = True

    # Create a mock close message
    close_msg = MagicMock(spec=WSMessage)
    close_msg.type = WSMsgType.CLOSE

    # Set up ws.__aiter__ to return our close message
    async def mock_aiter():
        yield close_msg
        # Prevent infinite loop
        raise asyncio.CancelledError()
    transport.ws.__aiter__ = mock_aiter

    transport._handle_disconnect = AsyncMock()
    transport.connect = AsyncMock()

    # Run _listen()
    with pytest.raises(asyncio.CancelledError):
        await transport._listen()

    # Verify proper handling
    assert transport._handle_disconnect.called
    assert transport.connect.called

@pytest.mark.asyncio
async def test_handle_disconnect_cleanup(transport):
    """Test that _handle_disconnect() properly cleans up the connection state."""
    # Set up initial connection state
    transport.ws = AsyncMock(spec=ClientWebSocketResponse)
    transport._connected = True
    transport.heartbeat = AsyncMock()

    # Call _handle_disconnect
    await transport._handle_disconnect()

    # Verify proper cleanup
    assert transport.heartbeat.stop.called
    assert transport.ws.close.called
    assert not transport._connected

@pytest.mark.asyncio
async def test_reconnection_after_error(transport):
    """Test the full reconnection cycle after a connection error."""
    # Mock dependencies
    transport.ws = AsyncMock(spec=ClientWebSocketResponse)
    transport._connected = True
    transport.heartbeat = AsyncMock()

    # Simulate a failed message send
    transport.ws.send_str = AsyncMock(side_effect=Exception("Cannot write to closing transport"))

    # Attempt to send a message
    with pytest.raises(Exception):
        await transport.send("test message")

    # Verify reconnection attempt
    assert not transport._connected  # Connection should be marked as disconnected
    assert transport.heartbeat.stop.called  # Heartbeat should be stopped
    assert transport.ws.close.called  # WebSocket should be closed
