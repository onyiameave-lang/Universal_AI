"""
shared.protocol.message
========================
Versioned, JSON-serializable message envelope for cross-repository
agent communication. This is the *only* structure agents are allowed
to put on the bus.

Design rules:
    * Frozen dataclass -- once emitted, a message is immutable.
    * JSON safe        -- only primitive types in to_dict().
    * Self-describing -- every field has a sensible default so a
                          new field can be added without breaking
                          older agents.
    * Bounded TTL      -- the bus can drop expired messages.
    * Correlation IDs  -- request/response pairs can be matched.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---- protocol version ----------------------------------------------------
PROTOCOL_VERSION = "1.0.0"


# ---- enums ---------------------------------------------------------------
class MessagePriority:
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageType:
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    AUDIT = "audit"
    PROPOSAL = "proposal"
    RETIREMENT = "retirement"


# ---- exceptions ----------------------------------------------------------
class ProtocolError(Exception):
    """Base class for protocol-level errors."""


class MessageValidationError(ProtocolError):
    """Raised when a message envelope is missing required fields."""


# ---- required fields -----------------------------------------------------
_REQUIRED = ("sender", "receiver", "task")
_KNOWN_TYPES = {MessageType.REQUEST, MessageType.RESPONSE, MessageType.EVENT,
                MessageType.BROADCAST, MessageType.HEARTBEAT, MessageType.AUDIT,
                MessageType.PROPOSAL, MessageType.RETIREMENT}
_KNOWN_CHANNELS = {
    "ecosystem.health", "ecosystem.status", "ecosystem.audit",
    "memory.query", "memory.store",
    "trading.signal", "trading.audit",
    "news.event", "social.event",
    "agent.proposal", "agent.retire",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# The envelope.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Message:
    """Immutable, JSON-serializable agent message.

    Required: sender, receiver, task.
    Optional: everything else has a sensible default.
    """
    sender: str
    receiver: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = MessagePriority.NORMAL
    type: str = MessageType.REQUEST
    channel: str = ""
    timestamp: str = field(default_factory=_now_iso)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    memory_reference: Optional[str] = None
    ttl_ms: Optional[int] = None
    protocol_version: str = PROTOCOL_VERSION
    source_repo: str = ""
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    status: str = "pending"
    confidence: Optional[float] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    memory_references: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        for f in _REQUIRED:
            if not getattr(self, f):
                raise MessageValidationError(
                    f"Message field {f!r} is required and cannot be empty"
                )
        if self.type not in _KNOWN_TYPES:
            raise MessageValidationError(
                f"Unknown message type {self.type!r}. "
                f"Expected one of {sorted(_KNOWN_TYPES)}"
            )
        if self.priority not in (0, 1, 2, 3):
            raise MessageValidationError(
                f"Priority must be 0..3, got {self.priority!r}"
            )

    # ----- serialization ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

    # ----- helpers ------------------------------------------------------
    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        if self.ttl_ms is None:
            return False
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        try:
            ts = int(
                datetime.fromisoformat(self.timestamp)
                .replace(tzinfo=timezone.utc)
                .timestamp() * 1000
            )
        except Exception:
            return False
        return (now - ts) > self.ttl_ms

    def reply(self, **kwargs: Any) -> "Message":
        """Return a response message addressed back to the original sender."""
        base = dict(
            sender=self.receiver,
            receiver=self.sender,
            task=f"{self.task}.reply",
            context={},
            type=MessageType.RESPONSE,
            correlation_id=self.message_id,
            reply_to=self.message_id,
            channel=self.channel,
            source_repo=self.source_repo,
        )
        base.update(kwargs)
        return Message(**base)


# --------------------------------------------------------------------------
# Convenience constructors.
# --------------------------------------------------------------------------
def request(sender: str, receiver: str, task: str, **kw: Any) -> Message:
    return Message(
        sender=sender, receiver=receiver, task=task, type=MessageType.REQUEST, **kw
    )


def event(sender: str, task: str, **kw: Any) -> Message:
    kw.pop("receiver", None)
    return Message(
        sender=sender, receiver="*", task=task, type=MessageType.EVENT, **kw
    )


def broadcast(sender: str, task: str, **kw: Any) -> Message:
    kw.pop("receiver", None)
    return Message(
        sender=sender, receiver="*", task=task, type=MessageType.BROADCAST, **kw
    )


def heartbeat(sender: str, **kw: Any) -> Message:
    return Message(
        sender=sender,
        receiver="coordinator",
        task="heartbeat",
        type=MessageType.HEARTBEAT,
        channel="ecosystem.health",
        ttl_ms=5000,
        **kw,
    )
