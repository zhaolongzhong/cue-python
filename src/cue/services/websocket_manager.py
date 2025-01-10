import json
import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Callable, Optional, Awaitable
from datetime import datetime
from dataclasses import dataclass

from tenacity import (
    retry,
    before_sleep_log,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from .transport import WebSocketTransport
from ..schemas.event_message import EventMessage

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    last_connected_at: Optional[datetime] = None
    last_disconnected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    connection_attempts: int = 0
    successful_messages_sent: int = 0
    failed_messages: int = 0
    last_close_reason: Optional[str] = None


class WebSocketState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


class WebSocketManager:
    """Manages WebSocket connection lifecycle and message handling with improved stability"""

    def __init__(
        self,
        ws_transport: WebSocketTransport,
        message_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_interval: float = 1.0,
        message_queue_size: int = 1000,
    ):
        self._transport = ws_transport
        self._state = WebSocketState.DISCONNECTED
        self._message_handlers = message_handlers or {}
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_interval = reconnect_interval
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._connection_lock = asyncio.Lock()
        self._metrics = ConnectionMetrics()

        # Message queue for handling backpressure
        self._message_queue = asyncio.Queue(maxsize=message_queue_size)
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    @property
    def is_connected(self) -> bool:
        return self._state == WebSocketState.CONNECTED

    @property
    def metrics(self) -> ConnectionMetrics:
        return self._metrics

    async def connect(self) -> None:
        """Establish WebSocket connection with enhanced error handling"""
        async with self._connection_lock:
            if self._state in (WebSocketState.CONNECTED, WebSocketState.CONNECTING):
                logger.debug(f"Connection attempt skipped - current state: {self._state}")
                return

            self._state = WebSocketState.CONNECTING
            self._metrics.connection_attempts += 1

            try:
                await self._transport.connect()
                self._state = WebSocketState.CONNECTED
                self._metrics.last_connected_at = datetime.now()
                self._start_message_processing()
                self._ensure_queue_processor()
                logger.debug(f"WebSocket connection established (attempt {self._metrics.connection_attempts})")
            except Exception as e:
                self._state = WebSocketState.DISCONNECTED
                self._metrics.last_error = str(e)
                logger.error(f"Failed to establish WebSocket connection: {e}")
                raise

    async def disconnect(self) -> None:
        """Gracefully close WebSocket connection with proper cleanup"""
        async with self._connection_lock:
            if self._state == WebSocketState.DISCONNECTED:
                return

            previous_state = self._state
            self._state = WebSocketState.CLOSING
            self._shutdown_event.set()  # Signal queue processor to stop

            try:
                # Cancel and wait for background tasks
                for task in [self._receive_task, self._reconnect_task, self._queue_processor_task]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Wait for pending messages with timeout
                try:
                    await asyncio.wait_for(self._message_queue.join(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for message queue to empty")

                await self._transport.disconnect()
                self._metrics.last_disconnected_at = datetime.now()
                logger.info(f"WebSocket disconnected (from state: {previous_state})")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self._state = WebSocketState.DISCONNECTED

    def _ensure_queue_processor(self) -> None:
        """Ensure the queue processor is running"""
        if not self._queue_processor_task or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_message_queue())

    async def _process_message_queue(self) -> None:
        """Process queued messages with backpressure handling"""
        while not self._shutdown_event.is_set():
            try:
                message = await self._message_queue.get()
                try:
                    if self._state == WebSocketState.CLOSING:
                        logger.warning("Queue processor: connection closing, message dropped")
                        continue

                    if not self.is_connected:
                        await self.connect()

                    await self._transport.send(message)
                    self._metrics.successful_messages_sent += 1
                except Exception as e:
                    self._metrics.failed_messages += 1
                    logger.error(f"Failed to send queued message: {e}")
                    if self._auto_reconnect and not self._shutdown_event.is_set():
                        await self._handle_connection_failure()
                finally:
                    self._message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in queue processor: {e}")

    def _start_message_processing(self) -> None:
        """Start background task for processing incoming messages"""
        if self._receive_task and not self._receive_task.done():
            return
        self._receive_task = asyncio.create_task(self._process_messages())

    async def _process_messages(self) -> None:
        """Process incoming messages with enhanced error handling"""
        while self._state == WebSocketState.CONNECTED:
            try:
                message = await self._transport.receive()

                if isinstance(message, dict):
                    message_type = message.get("type")
                    if message_type in self._message_handlers:
                        try:
                            event = EventMessage(**message)
                            await self._message_handlers[message_type](event)
                        except Exception as e:
                            logger.error(
                                f"Error in message handler for type {message_type}: {e}, message: {json.dumps(message, indent=4)}"
                            )
                    else:
                        logger.warning(f"No handler for message type: {message_type}")
                else:
                    logger.warning(f"Received non-dict message: {type(message)}")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                if "Cannot write to closing transport" in str(e):
                    logger.info("Transport closing, stopping message processing")
                    break
                if self._auto_reconnect and not self._shutdown_event.is_set():
                    await self._handle_connection_failure()
                else:
                    break

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _handle_connection_failure(self) -> None:
        """Handle connection failures with exponential backoff and metrics"""
        if self._state == WebSocketState.CLOSING or self._shutdown_event.is_set():
            logger.info("Connection failure handler: shutdown in progress, skipping reconnect")
            return

        self._state = WebSocketState.RECONNECTING
        try:
            await self._transport.connect()
            self._state = WebSocketState.CONNECTED
            self._metrics.last_connected_at = datetime.now()
            self._start_message_processing()
            logger.info("Successfully reconnected to WebSocket")
        except Exception as e:
            self._state = WebSocketState.DISCONNECTED
            self._metrics.last_error = str(e)
            logger.error(f"Failed to reconnect: {e}")
            raise

    async def send_message(self, message: str) -> None:
        """Queue a message for sending with backpressure handling"""
        try:
            await self._message_queue.put(message)
        except asyncio.QueueFull:
            logger.error("Message queue full, message dropped")
            self._metrics.failed_messages += 1
            raise RuntimeError("Message queue full")
