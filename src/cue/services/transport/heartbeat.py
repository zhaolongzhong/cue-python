import asyncio
import logging
from typing import Optional
from datetime import datetime

from .websocket_connection_error import WebSocketConnectionError

logger = logging.getLogger(__name__)


class WebSocketHeartbeat:
    def __init__(
        self,
        ws_transport: "AioHTTPWebSocketTransport",  # Forward reference  # noqa: F821
        heartbeat_interval: float = 60.0,  # Ping every 60 seconds
        heartbeat_timeout: float = 20.0,  # Wait 20 seconds for pong
        max_missed_heartbeats: int = 3,  # Reconnect after 3 missed heartbeats
    ):
        self.ws_transport = ws_transport
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_missed_heartbeats = max_missed_heartbeats
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_pong: Optional[datetime] = None
        self._missed_heartbeats = 0
        self._running = False

        # Event to track pong reception
        self._pong_event = asyncio.Event()

        # Flag to track if a ping has been sent
        self._ping_sent = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the heartbeat mechanism"""
        if self._running:
            return

        self._running = True
        self._last_pong = datetime.now()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug("WebSocket heartbeat started")

    async def stop(self):
        """Stop the heartbeat mechanism"""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket heartbeat stopped")

    async def _heartbeat_loop(self):
        """Main heartbeat loop using protocol-level ping/pong"""
        while self._running:
            try:
                async with self._lock:
                    # Clear the pong_event before sending ping
                    self._pong_event.clear()

                    # Set the ping_sent flag before sending ping
                    self._ping_sent = True

                    # Send protocol-level ping with unique Ping ID
                    await self.ws_transport.ping()

                try:
                    # Wait for pong within heartbeat_timeout
                    await asyncio.wait_for(self._pong_event.wait(), timeout=self.heartbeat_timeout)
                    # Pong received
                    self._missed_heartbeats = 0
                except asyncio.TimeoutError:
                    async with self._lock:
                        self._missed_heartbeats += 1
                        logger.warning(f"Missed heartbeat {self._missed_heartbeats}/{self.max_missed_heartbeats}")

                        if self._missed_heartbeats >= self.max_missed_heartbeats:
                            logger.error("Maximum missed heartbeats reached, reconnecting...")
                            await self.ws_transport.disconnect()
                            await self.ws_transport.connect()
                            self._missed_heartbeats = 0
                            self._last_pong = datetime.now()
                finally:
                    async with self._lock:
                        self._ping_sent = False

                # Wait for the heartbeat interval before sending the next ping
                await asyncio.sleep(self.heartbeat_interval)

            except WebSocketConnectionError as e:
                logger.error(f"WebSocket connection error during heartbeat: {e}")
                # Attempt to reconnect
                await asyncio.sleep(self.heartbeat_interval)
                await self.ws_transport.disconnect()
                await self.ws_transport.connect()
                async with self._lock:
                    self._missed_heartbeats = 0
                    self._last_pong = datetime.now()
                    self._ping_sent = False

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)

    def pong_received(self):
        """Call this method when a pong message is received"""

        async def _handle_pong():
            async with self._lock:
                if self._ping_sent:
                    self._last_pong = datetime.now()
                    self._missed_heartbeats = 0
                    self._pong_event.set()
                    self._ping_sent = False
                else:
                    logger.debug("Pong received but no ping was sent. Ignoring.")

        asyncio.create_task(_handle_pong())
