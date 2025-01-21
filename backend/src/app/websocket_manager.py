import json
import uuid
import asyncio
import logging
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from fastapi import WebSocket

from app.schemas.event_message import EventMessage, EventMessageType, ClientEventPayload

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class Connection:
    """Represents a websocket connection with metadata"""

    websocket: WebSocket
    participant_id: str
    client_id: str
    user_id: str
    connected_at: datetime
    state: ConnectionState = ConnectionState.PENDING
    last_ping: datetime = None
    cleanup_initiated: bool = False


class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, Connection] = {}
        self.client_sessions: dict[str, str] = {}
        self.user_sessions: dict[str, set[str]] = {}  # participant_id -> set of session_ids
        self.relationships: dict[str, set[str]] = {}  # user_id -> set of related ids
        self.closed_sessions: set[str] = set()
        self.lock = asyncio.Lock()

    def _create_session_id(self, client_id: str) -> str:
        """Create a unique session ID that includes client information"""
        return f"conn_{client_id}_{uuid.uuid4().hex[:8]}"

    def _get_state_snapshot(self) -> dict:
        """Get current state for debugging/logging"""
        return {
            "connections": len(self.connections),
            "clients": len(self.client_sessions),
            "users": {user_id: list(sessions) for user_id, sessions in self.user_sessions.items()},
            "relationships": {user_id: list(related) for user_id, related in self.relationships.items()},
        }

    async def connect(
        self,
        client_id: str,
        user_id: str,
        participant_id: str,
        websocket: WebSocket,
    ) -> str:
        """Establish new connection with relationship management"""
        session_id = self._create_session_id(client_id)
        logger.info(f"Starting connection for client:{client_id} user:{participant_id}")

        old_websocket = None
        old_session_id = None

        async with self.lock:
            # Get old session and mark it for cleanup if exists
            if old_session_id := self.client_sessions.get(client_id):
                if old_conn := self.connections.get(old_session_id):
                    if old_conn.state == ConnectionState.ACTIVE:
                        logger.info(
                            f"Found active connection {old_session_id} for client {client_id}, marking for cleanup"
                        )
                        old_websocket = old_conn.websocket
                        await self._cleanup_connection(old_session_id)

            # Create new connection
            conn = Connection(
                websocket=websocket,
                participant_id=participant_id,
                client_id=client_id,
                user_id=user_id,
                state=ConnectionState.PENDING,
                connected_at=datetime.now(timezone.utc),
            )
            self.connections[session_id] = conn
            self.client_sessions[client_id] = session_id

            # Update user sessions
            if participant_id not in self.user_sessions:
                self.user_sessions[participant_id] = set()
            self.user_sessions[participant_id].add(session_id)

            # Handle relationships
            if user_id not in self.relationships:
                self.relationships[user_id] = set()
            self.relationships[user_id].add(participant_id)

            # Mark connection as active
            conn.state = ConnectionState.ACTIVE
            state = self._get_state_snapshot()
            logger.info(f"Connection established - State: {json.dumps(state, indent=4)}")

        # Close old websocket outside of lock if it exists
        if old_websocket and not old_websocket.client_state.DISCONNECTED:
            try:
                logger.info(f"Closing old websocket for {old_session_id}")
                await old_websocket.close(code=1000)
            except Exception as e:
                logger.info(f"Error closing old websocket: {e}")

        logger.info(f"Participant '{participant_id}' connected with session_id '{session_id}'")
        return session_id

    async def _cleanup_connection(self, session_id: str, initiated_by_disconnect: bool = False):
        """Internal method to clean up a connection"""

        if session_id in self.closed_sessions:
            logger.info(f"Session {session_id} already cleaned up, skipping")
            return

        if conn := self.connections.get(session_id):
            if conn.cleanup_initiated and not initiated_by_disconnect:
                logger.info(f"Cleanup already initiated for {session_id}, skipping")
                return

            logger.info(f"Starting cleanup for session: {session_id}, client_id: {conn.client_id}")
            conn.cleanup_initiated = True

            # Remove from mappings first
            self.connections.pop(session_id)
            self.client_sessions.pop(conn.client_id, None)

            if conn.participant_id in self.user_sessions:
                self.user_sessions[conn.participant_id].discard(session_id)
                if not self.user_sessions[conn.participant_id]:
                    del self.user_sessions[conn.participant_id]

            # Try to close websocket
            try:
                if not conn.websocket.client_state.DISCONNECTED:
                    logger.info(f"Explicitly closing websocket for session: {session_id}")
                    await conn.websocket.close(code=1000)
            except Exception as e:
                logger.info(f"Websocket close result for {session_id}: {e}")

            # Mark as closed
            self.closed_sessions.add(session_id)

            state = self._get_state_snapshot()
            logger.info(f"Cleanup completed for session: {session_id}. State: {json.dumps(state, indent=4)}")

    async def disconnect(self, session_id: str):
        """
        Handle disconnection while preserving relationships if there are active assistants.
        For assistants, we always clean up relationships.
        """

        if session_id in self.closed_sessions:
            logger.info(f"Session {session_id} already handled by cleanup, skipping disconnect")
            return set()

        async with self.lock:
            # Find the participant_id and gather info while holding the lock
            # keep track other ids to notify disconnect event

            if conn := self.connections.get(session_id):
                if conn.state not in (ConnectionState.ACTIVE, ConnectionState.PENDING):
                    logger.info(f"Session {session_id} already in {conn.state} state, skipping disconnect")
                    return set()
                logger.info(f"Disconnecting {conn.client_id} ({conn.participant_id})")

                # Get related participants before cleanup
                other_ids = self.relationships.get(conn.participant_id, set()).copy()

                await self._cleanup_connection(session_id)

                # Check if user has any remaining sessions
                if conn.participant_id not in self.user_sessions:
                    if conn.user_id in self.relationships:
                        participants = self.relationships.get(conn.user_id, set())
                        if participants:
                            if conn.participant_id in participants:
                                participants.discard(conn.participant_id)
                            if not participants and conn.user_id in self.relationships:
                                del self.relationships[conn.user_id]

                state = self._get_state_snapshot()
                logger.info(f"Disconnect complete - State: {json.dumps(state, indent=4)}")

                # Return related participants for notification
                return other_ids

    async def get_participants(self, participant_id: str, active_only: bool = False) -> set[str]:
        """
        Get all participants related to the given participant_id.

        Args:
            participant_id: The ID of the participant to get relations for
            active_only: If True, only return active (connected) participants

        Returns:
            Set of related participant IDs
        """
        async with self.lock:
            # Get all related participants
            related = self.relationships.get(participant_id, set()).copy()

            # If we only want active participants, filter out inactive ones
            if active_only:
                related = {pid for pid in related if pid in self.user_sessions and self.user_sessions[pid]}
            return related

    async def send_personal_message(self, message: str, session_id: str):
        """Send a message to a specific session without acquiring the main lock."""
        conn = self.connections.get(session_id)
        if conn.websocket:
            try:
                await conn.websocket.send_text(message)
                logger.debug(f"Sent message to session_id '{session_id}'")
            except Exception as e:
                logger.error(f"Error sending message to session_id '{session_id}': {e}")
        else:
            logger.error(f"send_personal_message websocket is none, session_id: {session_id}")

    async def broadcast_message(self, message: str, user_id: str) -> tuple[bool, Optional[str]]:
        """Broadcast a message to all sessions of a user."""
        sessions = set()

        async with self.lock:
            if user_id in self.user_sessions:
                sessions = self.user_sessions[user_id].copy()

        if not sessions:
            msg = f"Participant '{user_id}' is offline, active_users: {self.user_sessions}, sessions: {sessions}, message: {message}"
            logger.info(msg)
            state = self._get_state_snapshot()
            logger.info(f"broadcast_message - State: {json.dumps(state, indent=4)}")
            return False, msg

        for session_id in sessions:
            await self.send_personal_message(message, session_id)
        return True, None

    async def can_message(self, sender_id: str, recipient_id: str) -> bool:
        """Check if sender can message recipient based on their relationship."""
        async with self.lock:
            if "all" in recipient_id:
                return True
            return recipient_id in self.relationships.get(sender_id, set())

    async def send_message(self, message: EventMessage) -> list[tuple[bool, Optional[str]]]:
        """Send message if allowed by relationship."""
        payload = message.payload
        sender_user_id = payload.user_id
        recipient = payload.recipient
        if not await self.can_message(sender_user_id, recipient):
            msg = f"Message blocked: '{sender_user_id}' cannot message '{recipient}'"
            logger.warning(msg)
            return [(False, msg)]

        results = []
        if "all" in recipient:
            recipients = self.relationships.get(sender_user_id, set())
            for recipient in recipients:
                status, msg = await self.broadcast_message(message=message.model_dump_json(), user_id=recipient)
                results.append((status, msg))
        else:
            status, msg = await self.broadcast_message(message=message.model_dump_json(), user_id=recipient)
            results.append((status, msg))
        return results

    async def broadcast_connection_event(
        self, client_id: str, sender_id: str, recipients: Optional[list[str]] = None, is_connect: bool = True
    ):
        """
        Broadcasts connection/disconnection events to specified recipients.

        Args:
            client_id (str): ID of the client triggering the event
            sender_id (str): ID of the sender
            recipients (list[str]): List of recipient IDs
            is_connect (bool): True for connect event, False for disconnect
        """
        if not recipients:
            return
        event_type = EventMessageType.CLIENT_CONNECT if is_connect else EventMessageType.CLIENT_DISCONNECT
        action_state = "connected" if is_connect else "disconnected"

        event_message = EventMessage(
            type=event_type,
            client_id=client_id,
            payload=ClientEventPayload(
                client_id=client_id,
                sender=sender_id,
                message=f"Participant '{sender_id}' {action_state} successfully: active_connections: {self.connections}, active_users: {self.user_sessions}, relationship: {self.relationships} ",
            ),
        )
        message_text = event_message.model_dump_json()

        for recipient in recipients:
            await self.broadcast_message(message_text, recipient)

    async def get_user_sessions(self, user_id: str) -> set[str]:
        """Get all active sessions for a user."""
        async with self.lock:
            return self.user_sessions.get(user_id, set()).copy()
