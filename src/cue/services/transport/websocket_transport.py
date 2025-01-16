import json
import random
import asyncio
import logging
from typing import Any, Dict, Optional, Protocol
from typing_extensions import runtime_checkable

import aiohttp
from aiohttp.client_ws import WSMsgType, ClientWSTimeout

from .heartbeat import WebSocketHeartbeat
from .websocket_connection_error import WebSocketConnectionError

logger = logging.getLogger(__name__)


@runtime_checkable
class WebSocketTransport(Protocol):
    """Protocol for WebSocket transport operations"""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def send(self, message: str) -> None: ...

    async def receive(self) -> Dict[str, Any]: ...

    async def ping(self) -> None: ...


class AioHTTPWebSocketTransport(WebSocketTransport):
    """AIOHTTP implementation of WebSocket transport with protocol-level ping/pong and single receiver."""

    def __init__(
        self,
        ws_url: str,
        client_id: str,
        api_key: str,
        runner_id: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ):
        self.ws_url = ws_url
        self.session = session or aiohttp.ClientSession()
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.client_id = client_id
        self.api_key = api_key
        self.runner_id = runner_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False
        self.heartbeat = WebSocketHeartbeat(
            self,
            heartbeat_interval=60.0,  # Ping every 60 seconds
            heartbeat_timeout=20.0,  # Wait 20 seconds for pong
            max_missed_heartbeats=3,  # Reconnect after 3 missed heartbeats
        )

        # Initialize the message queue
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, start_listener: bool = True) -> None:
        """Establish WebSocket connection with retry logic and proper error handling"""
        if self._connected and self.ws and not self.ws.closed:
            logger.debug("WebSocket is already connected.")
            return

        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {
                    "X-API-Key": f"{self.api_key}",
                    "Connection": "upgrade",
                    "Upgrade": "websocket",
                    "Sec-WebSocket-Version": "13",
                }

                ws_url_with_params = f"{self.ws_url}/{self.client_id}"
                if self.runner_id:
                    ws_url_with_params += f"?runner_id={self.runner_id}"
                self.ws = await self.session.ws_connect(
                    ws_url_with_params,
                    headers=headers,
                    heartbeat=None,  # disable AioHTTP's built-in heartbeat
                    timeout=ClientWSTimeout(ws_close=30.0),
                )

                self._connected = True
                logger.debug(f"WebSocket connection established for client {self.client_id}")

                # Start heartbeat after connection is established
                await self.heartbeat.start()

                # Start listening to incoming messages
                if start_listener:
                    asyncio.create_task(self._listen())
                return

            except aiohttp.ClientResponseError as e:
                if e.status == 401:
                    logger.error("Authentication failed: Invalid or expired api key")
                    raise WebSocketConnectionError("Authentication failed: Please check your api key")
                logger.error(f"HTTP error during WebSocket connection: {e.status} - {e.message}")

            except aiohttp.WSServerHandshakeError as e:
                logger.error(f"WebSocket handshake failed: {str(e)}")
                if attempt == self.max_retries:
                    raise WebSocketConnectionError(f"WebSocket handshake failed after {self.max_retries} attempts")

            except aiohttp.ClientError as e:
                logger.error(f"Connection error: {str(e)}")
                if attempt == self.max_retries:
                    raise WebSocketConnectionError(f"Failed to establish WebSocket connection: {str(e)}")

            except Exception as e:
                logger.error(f"Unexpected error during WebSocket connection: {str(e)}")
                raise WebSocketConnectionError(f"Unexpected error: {str(e)}")

            # Exponential backoff
            backoff = self.retry_delay * (2 ** (attempt - 1))
            logger.debug(f"Retrying WebSocket connection in {backoff} seconds...")
            await asyncio.sleep(backoff)

    async def _listen(self):
        """Listen for incoming messages, handle pongs, and auto-reconnect on disconnect."""
        while True:
            try:
                async for msg in self.ws:
                    if msg.type == WSMsgType.PONG:
                        logger.debug("Protocol-level pong received from _listen")
                        self.heartbeat.pong_received()
                    elif msg.type == WSMsgType.PING:
                        logger.debug("Protocol-level ping received, sending pong")
                        await self.ws.pong()
                    elif msg.type == WSMsgType.TEXT:
                        data = msg.data
                        logger.debug(f"Received message: {data}")
                        await self._message_queue.put(data)
                    elif msg.type == WSMsgType.CLOSE:
                        logger.info("WebSocket connection closed by server")
                        break
                    elif msg.type == WSMsgType.ERROR:
                        logger.error("WebSocket connection error")
                        break
            except Exception as e:
                logger.error(f"Error in WebSocket listen loop: {str(e)}")

            logger.info("WebSocket listener terminated, attempting to reconnect")
            try:
                await self.reconnect()
            except Exception as reconnect_error:
                logger.error(f"Reconnection failed, exiting listen loop: {reconnect_error}")
                break

    async def disconnect(self) -> None:
        """Safely close the WebSocket connection"""
        if self.ws and not self.ws.closed:
            try:
                await self.heartbeat.stop()
                await self.ws.close()
                self._connected = False
                logger.info(f"WebSocket connection closed for client {self.client_id}")
            except Exception as e:
                logger.error(f"Error during WebSocket disconnection: {str(e)}")

    async def send(self, message: str) -> None:
        """Send message with connection check and error handling"""
        try:
            if not self._connected or not self.ws or self.ws.closed:
                logger.debug("WebSocket not connected. Attempting to connect before sending.")
                await self.connect()
            await self.ws.send_str(message)
            logger.debug(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise WebSocketConnectionError(f"Failed to send message: {str(e)}")

    async def receive(self) -> Dict[str, Any]:
        """Receive message with connection check and error handling"""
        data = await self._message_queue.get()
        try:
            # Instead of calling self.ws.receive(), get messages from the queue
            payload = json.loads(data)
            if payload.get("error") and payload.get("code") == 429:
                logger.warning("Rate limit hit.")
                await asyncio.sleep(1)
                return None
            return payload
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise WebSocketConnectionError(f"Invalid JSON message received: {str(e)}")
        except Exception as e:
            logger.error(f"Error receiving message: {str(e)}")
            raise WebSocketConnectionError(f"Failed to receive message: {str(e)}")

    async def ping(self) -> None:
        """Send a protocol-level ping and await pong"""
        try:
            pong_waiter = self.ws.ping()  # Initiates a ping and returns a future for the pong
            # logger.debug("Ping")
            await asyncio.wait_for(pong_waiter, timeout=self.heartbeat.heartbeat_timeout)
            # logger.debug("Pong")
            self.heartbeat.pong_received()
        except asyncio.TimeoutError:
            logger.error("Protocol-level pong not received within timeout")
            raise WebSocketConnectionError("Pong not received in response to ping")
        except Exception as e:
            logger.error(f"Error during protocol-level ping: {str(e)}")
            raise WebSocketConnectionError(f"Ping failed: {str(e)}")

    async def reconnect(self) -> None:
        backoff = self.retry_delay
        MAX_BACKOFF_LIMIT = 300  # maximum 5 minutes
        JITTER_FACTOR = 0.1  # ±10% jitter
        while True:
            try:
                logger.info("Attempting to reconnect...")
                await self.disconnect()
                await self.connect(start_listener=False)
                logger.info("Reconnected successfully")
                return
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")
                # Apply jitter: randomize backoff slightly to avoid thundering herd
                jitter = backoff * JITTER_FACTOR * (random.random() * 2 - 1)
                sleep_time = min(backoff + jitter, MAX_BACKOFF_LIMIT)
                logger.debug(f"Waiting {sleep_time:.2f} seconds before next attempt...")
                await asyncio.sleep(sleep_time)
                # Increase backoff, capped at MAX_BACKOFF_LIMIT
                backoff = min(backoff * 2, MAX_BACKOFF_LIMIT)
