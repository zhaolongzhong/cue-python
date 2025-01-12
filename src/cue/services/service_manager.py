import json
import uuid
import asyncio
import logging
import platform
from typing import Union, Callable, Optional, Awaitable

import aiohttp
from aiohttp import ClientResponseError, ClientConnectionError
from pydantic import BaseModel

from ..config import get_settings
from ..schemas import (
    Assistant,
    AgentConfig,
    FeatureFlag,
    RunMetadata,
    MessageParam,
    CompletionResponse,
    ToolResponseWrapper,
)
from .transport import (
    AioHTTPTransport,
    AioHTTPWebSocketTransport,
)
from .memory_client import MemoryClient
from .message_client import MessageClient
from .assistant_client import AssistantClient
from .monitoring_client import MonitoringClient
from .websocket_manager import WebSocketManager
from .conversation_client import ConversationClient
from ..schemas.event_message import EventMessage, MessagePayload, EventMessageType, ClientEventPayload

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manager for coordinating service clients and handling WebSocket interactions."""

    def __init__(
        self,
        run_metadata: RunMetadata,
        feature_flag: FeatureFlag,
        base_url: str,
        session: aiohttp.ClientSession,
        on_message_received: Callable[[dict[str, any]], Awaitable[None]] = None,
        agent: Optional[AgentConfig] = None,
    ):
        self.run_metadata = run_metadata
        self.feature_flag = feature_flag
        self.base_url = base_url
        self.api_key = agent.api_key if agent else None
        self.agent = agent
        self.overwrite_agent_config: Optional[AgentConfig] = None
        self.assistant_id: Optional[str] = agent.id if agent else None
        self.client_id: Optional[str] = agent.client_id if agent else run_metadata.id
        self.user_id: Optional[str] = None
        self.is_server_available = False
        if platform.system() != "Darwin" and "http://localhost" in self.base_url:
            self.base_url = self.base_url.replace("http://localhost", "http://host.docker.internal")
        self._session = session
        self.on_message_received = on_message_received
        self.recipient: Optional[str] = None  # either user id or assistant runner id
        self.runner_id: Optional[str] = None
        if self.run_metadata.mode != "client":
            # it will be used as participant so we can find the websocket session of this client by using this id
            self.runner_id = self.client_id

        self._http = AioHTTPTransport(base_url=self.base_url, api_key=self.api_key, session=self._session)
        self._ws = AioHTTPWebSocketTransport(
            ws_url=self.base_url.replace("http", "ws") + "/ws",
            client_id=self.client_id,
            api_key=self.api_key,
            runner_id=self.runner_id if self.run_metadata.mode != "client" else None,
            session=self._session,
        )

        self._ws_manager = WebSocketManager(
            ws_transport=self._ws,
            message_handlers={
                "client_disconnect": self._handle_client_connect,
                "client_connect": self._handle_client_connect,
                "client_status": self._handle_client_connect,
                "ping": self._handle_ping,
                "pong": self._handle_pong,
                "generic": self._handle_generic,
                "message": self._handle_message,
                "user": self._handle_message,
                "assistant": self._handle_message,
                # Control event handlers
                "control_reset": self._handle_control_reset,
                "control_get_state": self._handle_control_get_state,
                "control_permission": self._handle_control_permission,
                "control_permission_response": self._handle_control_permission_response,
            },
        )

        # Initialize resource clients
        self.assistants = AssistantClient(self._http)
        self.memories = MemoryClient(self._http)
        self.conversations = ConversationClient(self._http)
        self.messages = MessageClient(self._http)
        self.monitoring = MonitoringClient(self._http)

    @classmethod
    async def create(
        cls,
        run_metadata: RunMetadata,
        feature_flag: Optional[FeatureFlag] = FeatureFlag(),
        base_url: Optional[str] = None,
        on_message_received: Callable[[dict[str, any]], Awaitable[None]] = None,
        agent: Optional[AgentConfig] = None,
    ):
        settings = get_settings()
        base_url = base_url or settings.API_URL
        session = aiohttp.ClientSession()
        service_manager = cls(
            run_metadata=run_metadata,
            feature_flag=feature_flag,
            base_url=base_url,
            session=session,
            on_message_received=on_message_received,
            agent=agent,
        )
        await service_manager.initialize()
        return service_manager

    async def initialize(self) -> None:
        try:
            self.is_server_available = await self._check_server_availability()
            self._http.is_server_available = self.is_server_available
            if not self.is_server_available:
                return
        except Exception as e:
            logger.error(f"Server availability check failed: {e}")
            return
        if self.feature_flag.enable_storage:
            await self._prepare_conversation(self.agent)

    async def close(self) -> None:
        """Close all connections"""
        await self._ws.disconnect()
        await self._ws_manager.disconnect()
        await self._session.close()

    async def connect(self) -> None:
        """Establish connection to the service"""

        if not self.is_server_available:
            logger.error("Server is not available.")
            return
        await self._ws_manager.connect()

    async def disconnect(self) -> None:
        """Close all connections"""
        await self._ws_manager.disconnect()
        await self._session.close()

    async def broadcast(self, message: str) -> None:
        if not self.is_server_available:
            return
        await self._ws_manager.send_message(message)

    async def send_message_to_assistant(self, message: str) -> None:
        websocket_request_id = str(uuid.uuid4())
        msg = EventMessage(
            type=EventMessageType.USER,
            payload=MessagePayload(
                message=message,
                recipient="all",
                websocket_request_id=websocket_request_id,
                metadata={"author": {"role": "user"}, "recipient": self.assistant_id},
            ),
            websocket_request_id=websocket_request_id,
        )
        await self.broadcast(msg.model_dump_json())

    async def send_message_to_user(self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]) -> None:
        payload = None
        role = "assistant"
        name = None
        model = None

        if isinstance(message, ToolResponseWrapper):
            role = "tool" if message.tool_messages else "user"
            name = message.author.name
            payload = message.model_dump(exclude=None, exclude_unset=True, exclude_defaults=True)
            model = message.model
        elif isinstance(message, CompletionResponse):
            if isinstance(message.response, BaseModel):
                payload = message.response.model_dump(exclude_none=True)
            name = self.assistant_id
            model = message.model
        elif isinstance(message, MessageParam):
            role = message.role
            name = message.name

        author = {"role": role}
        if name:
            author["name"] = name

        msg = EventMessage(
            type=EventMessageType.ASSISTANT,
            payload=MessagePayload(
                message=message.get_text(),
                sender=self.runner_id,
                recipient="",  # empty or user id
                payload=payload,
                websocket_request_id=str(uuid.uuid4()),
                metadata={"author": author, "model": model},
                msg_id=message.msg_id,
            ),
            websocket_request_id=str(uuid.uuid4()),
        )
        await self.broadcast(msg.model_dump_json())

    async def broadcast_client_status(self) -> None:
        logger.debug("broadcast client status")
        msg = EventMessage(
            type=EventMessageType.CLIENT_STATUS,
            client_id=self.client_id,
            payload=ClientEventPayload(
                client_id=self.client_id,
                sender="",
                recipient="",
                payload={"runner_id": self.runner_id, "assistant_id": self.assistant_id},
            ),
        )
        await self.broadcast(msg.model_dump_json())

    async def _handle_client_connect(self, message: EventMessage) -> None:
        if message.type.value == EventMessageType.CLIENT_CONNECT:
            self.user_id = message.payload.user_id
            await self.broadcast_client_status()

        logger.info(f"Client connect event {message.payload.client_id} event: {message.model_dump_json(indent=4)}")

    async def _handle_ping(self, message: EventMessage) -> None:
        logger.debug(f"Received ping message: {message}")

    async def _handle_pong(self, message: EventMessage) -> None:
        logger.debug(f"Received pong message: {message}.")
        self._ws.handle_pong(message=message)

    async def _handle_generic(self, message: EventMessage) -> None:
        logger.debug(f"Handling generic message: {message}")

    async def _handle_message(self, message: EventMessage) -> None:
        if self.on_message_received:
            try:
                await self.on_message_received(message)
            except Exception as e:
                # Catch any other errors in on_message handling
                logger.error(f"Error in on_message handling: {e}. Message content: {message.model_dump_json(indent=4)}")

    async def _check_server_availability(self) -> bool:
        """Check if the server is running by performing an HTTP GET request to the health endpoint."""
        health_url = f"{self.base_url}/health"
        try:
            async with self._session.get(health_url, timeout=10) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Health check failed with status {response.status}: {text}")
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Health check failed: {text}",
                        headers=response.headers,
                    )
                data = await response.json()
                if data.get("status") != "ok":
                    logger.error(f"Health check returned unexpected status: {data}")
                    raise ValueError(f"Unexpected health check status: {data.get('status')}")
                logger.debug("Server is available based on health check.")
                return True
        except asyncio.TimeoutError:
            logger.error("Health check request timed out.")
            raise
        except ClientConnectionError as e:
            logger.warning(f"Failed to connect to server for health check: {e}")
            return False
        except ClientResponseError as e:
            logger.error(f"Server responded with an error during health check: {e}")
        except json.JSONDecodeError:
            logger.error("Health check response is not valid JSON.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            raise

    async def _prepare_conversation(self, agent: AgentConfig):
        if not self.assistant_id:
            self.assistant_id = await self.assistants.create_default_assistant(agent.name)
        else:
            self.assistants._default_assistant_id = self.assistant_id
        if not self.assistant_id:
            raise
        self.memories.set_default_assistant_id(self.assistant_id)
        conversation_id = await self.conversations.create_default_conversation(self.assistant_id)
        self.messages.set_default_conversation_id(conversation_id)
        self.monitoring.set_context(assistant_id=self.assistant_id, conversation_id=conversation_id)
        await self._get_assistant(self.assistant_id)

        logger.debug(f"_prepare_conversation: {json.dumps(self.get_conversation_metadata(), indent=4)}")

    async def _get_assistant(self, id: str) -> Assistant:
        assistant = await self.assistants.get(self.assistant_id)
        if assistant.metadata and assistant.metadata.model:
            self.overwrite_agent_config = AgentConfig(model=assistant.metadata.model)

    def get_overwrite_model(self) -> Optional[str]:
        if self.overwrite_agent_config:
            return self.overwrite_agent_config.model
        return None

    def get_conversation_metadata(self) -> dict:
        medadata = {
            "assistant_id": self.assistant_id,
            "conversation_id": self.messages.default_conversation_id,
            "model": self.overwrite_agent_config.model if self.overwrite_agent_config else None,
        }
        return medadata

    async def _handle_control_reset(self, message: EventMessage) -> None:
        """Handle agent reset request"""
        logger.info(f"Received reset request: {message.model_dump_json(indent=4)}")
        reset_type = message.payload.reset_type
        try:
            if reset_type == "full":
                # Full reset - clear all state including conversation history
                await self.memories.clear()
                await self.conversations.create_default_conversation(self.assistant_id)
                self.overwrite_agent_config = None
            else:
                # Soft reset - clear current conversation state only
                await self.conversations.create_default_conversation(self.assistant_id)

            # Send confirmation
            msg = EventMessage(
                type=EventMessageType.CONTROL_STATE,
                payload=ControlStatePayload(
                    message="Reset completed successfully",
                    control_type="state",
                    state={"reset_type": reset_type, "status": "success"}
                )
            )
            await self.broadcast(msg.model_dump_json())
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            # Send error
            msg = EventMessage(
                type=EventMessageType.ERROR,
                payload=MessagePayload(
                    message=f"Reset failed: {str(e)}",
                    metadata={"error_type": "reset_failed"}
                )
            )
            await self.broadcast(msg.model_dump_json())

    async def _handle_control_get_state(self, message: EventMessage) -> None:
        """Handle request for agent state"""
        logger.info("Received state request")
        try:
            state = {
                "agent_id": self.assistant_id,
                "conversation_id": self.messages.default_conversation_id,
                "model": self.get_overwrite_model(),
                "metadata": self.get_conversation_metadata(),
                "websocket_status": {
                    "connected": self._ws_manager.is_connected,
                    "metrics": self._ws_manager.metrics.__dict__
                }
            }

            msg = EventMessage(
                type=EventMessageType.CONTROL_STATE,
                payload=ControlStatePayload(
                    control_type="state",
                    state=state
                )
            )
            await self.broadcast(msg.model_dump_json())
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            msg = EventMessage(
                type=EventMessageType.ERROR,
                payload=MessagePayload(
                    message=f"Failed to get state: {str(e)}",
                    metadata={"error_type": "state_retrieval_failed"}
                )
            )
            await self.broadcast(msg.model_dump_json())

    async def _handle_control_permission(self, message: EventMessage) -> None:
        """Handle permission request from assistant"""
        logger.info(f"Handling permission request: {message.model_dump_json(indent=4)}")
        # Forward the permission request to the user
        await self.broadcast(message.model_dump_json())

    async def _handle_control_permission_response(self, message: EventMessage) -> None:
        """Handle permission response from user"""
        logger.info(f"Received permission response: {message.model_dump_json(indent=4)}")
        if self.on_message_received:
            try:
                await self.on_message_received(message)
            except Exception as e:
                logger.error(f"Error handling permission response: {e}")

    async def request_permission(self, permission_type: str, reason: str, details: Optional[dict] = None) -> None:
        """Request permission from user"""
        msg = EventMessage(
            type=EventMessageType.CONTROL_PERMISSION,
            payload=ControlPermissionPayload(
                control_type="permission",
                permission_type=permission_type,
                reason=reason,
                details=details
            )
        )
        await self.broadcast(msg.model_dump_json())

    async def reset_agent(self, reset_type: str = "full") -> None:
        """Reset agent state"""
        msg = EventMessage(
            type=EventMessageType.CONTROL_RESET,
            payload=ControlResetPayload(
                control_type="reset",
                reset_type=reset_type
            )
        )
        await self.broadcast(msg.model_dump_json())

    async def get_agent_state(self) -> None:
        """Request current agent state"""
        msg = EventMessage(
            type=EventMessageType.CONTROL_GET_STATE,
            payload=ControlStatePayload(
                control_type="state"
            )
        )
        await self.broadcast(msg.model_dump_json())
