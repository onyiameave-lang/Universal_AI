"""
shared.protocol
================
Cross-repository agent communication protocol for the AI Ecosystem.

Every agent in UniversalAI, MemoryAI, MarketOracle, NewsIntel, and
SocialIntel communicates exclusively through the structures defined
here. The Coordinator in UniversalAI can therefore route, audit, and
orchestrate work without depending on the internals of any single
repository.

Public surface:
    Message           -- versioned, JSON-serializable envelope
    MessagePriority   -- 0=low, 1=normal, 2=high, 3=critical
    MessageType       -- request, response, event, broadcast
    MessageBus        -- in-process pub/sub + request/reply
    get_bus()         -- process-wide default bus
    AgentRegistry     -- directory of agents and their capabilities
    get_registry()    -- process-wide default registry
    AgentSpec         -- declarative description of an agent
    request/event/broadcast/heartbeat -- convenience constructors
"""
from shared.protocol.message import (
    Message,
    MessagePriority,
    MessageType,
    PROTOCOL_VERSION,
    ProtocolError,
    MessageValidationError,
    request,
    event,
    broadcast,
    heartbeat,
)
from shared.protocol.bus import MessageBus, get_bus, set_bus
from shared.protocol.registry import AgentRegistry, AgentSpec, get_registry

__all__ = [
    "Message",
    "MessagePriority",
    "MessageType",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "MessageValidationError",
    "MessageBus",
    "get_bus",
    "set_bus",
    "AgentRegistry",
    "AgentSpec",
    "get_registry",
    "request",
    "event",
    "broadcast",
    "heartbeat",
]

__version__ = PROTOCOL_VERSION
