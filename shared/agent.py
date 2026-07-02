"""
shared.agent
============
Abstract base class for all agents in the ecosystem.

Every agent:
    1. is registered with the AgentRegistry on start,
    2. subscribes to one or more channels on the bus,
    3. emits heartbeats on a configurable interval,
    4. answers tasks routed to it by the Coordinator,
    5. can be gracefully retired.

The base class is transport-agnostic -- the same class works with the
in-process bus, a Redis bus, or a future HTTP/gRPC bus.
"""
from __future__ import annotations

import abc
import logging
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from shared.protocol import (
    AgentSpec,
    Message,
    MessageType,
    heartbeat,
    get_bus,
    get_registry,
)

log = logging.getLogger("ecosystem.agent")


class BaseAgent(abc.ABC):
    """Base class every concrete agent inherits from."""

    name: str = "base"
    description: str = ""
    version: str = "1.0.0"
    repository: str = ""
    capabilities: List[str] = []
    channels: List[str] = []
    heartbeat_interval_sec: float = 10.0
    mission: Dict[str, Any] = {}
    domain: str = ""
    memory_namespace: str = ""
    security_level: str = "standard"
    lifecycle_status: str = "active"
    metadata: Dict[str, Any] = {}

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        self.name = name or kwargs.get("name") or self.name
        self.description = kwargs.get("description", self.description)
        self.version = kwargs.get("version", self.version)
        self.repository = kwargs.get("repository", self.repository)
        self.capabilities = kwargs.get("capabilities", self.capabilities)
        self.channels = kwargs.get("channels", self.channels)
        self.heartbeat_interval_sec = kwargs.get("heartbeat_interval_sec", self.heartbeat_interval_sec)
        self.mission = kwargs.get("mission", self.mission)
        self.domain = kwargs.get("domain", self.domain)
        self.memory_namespace = kwargs.get("memory_namespace", self.memory_namespace)
        self.security_level = kwargs.get("security_level", self.security_level)
        self.lifecycle_status = kwargs.get("lifecycle_status", self.lifecycle_status)
        self.metadata = kwargs.get("metadata", self.metadata)
        self._bus = get_bus()
        self._registry = get_registry()
        self._stop_event = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._tasks_handled = 0
        self._tasks_failed = 0
        self._last_activity_ts: float = 0.0
        self._lock = threading.RLock()

    # ----- lifecycle ----------------------------------------------------
    def start(self) -> None:
        for ch in self.channels:
            self._bus.subscribe(ch, self._on_message)
        spec = AgentSpec(
            name=self.name,
            capabilities=list(self.capabilities),
            repository=self.repository,
            description=self.description,
            version=self.version,
            accepts_channels=list(self.channels),
            mission=dict(self.mission or {}),
            domain=self.domain,
            memory_namespace=self.memory_namespace,
            security_level=self.security_level,
            lifecycle_status=self.lifecycle_status,
            metadata=dict(self.metadata or {}),
        )
        self._registry.register(spec, handler=self.handle)
        self._stop_event.clear()
        if self.heartbeat_interval_sec > 0:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True,
                name=f"hb-{self.name}")
            self._hb_thread.start()
        log.info("Agent %s started (caps=%s, channels=%s)",
                 self.name, self.capabilities, self.channels)

    def stop(self) -> None:
        self._stop_event.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2.0)
        for ch in self.channels:
            self._bus.unsubscribe(ch)
        self._registry.unregister(self.name)
        log.info("Agent %s stopped", self.name)

    # ----- handlers -----------------------------------------------------
    def _on_message(self, msg: Message) -> None:
        if msg.receiver in (self.name, "*") or msg.channel in self.channels:
            self._last_activity_ts = time.time()
            try:
                self.handle(msg)
                self._tasks_handled += 1
            except Exception as exc:
                self._tasks_failed += 1
                log.exception("Agent %s failed handling %s: %s",
                              self.name, msg.task, exc)
                self._bus.send(Message(
                    sender=self.name,
                    receiver="auditor",
                    task="agent.error",
                    type=MessageType.AUDIT,
                    context={
                        "agent": self.name,
                        "task": msg.task,
                        "trace": traceback.format_exc()[-4000:],
                    },
                    channel="ecosystem.audit",
                ))

    @abc.abstractmethod
    def handle(self, msg: Message) -> None:
        """Implement the agent's behavior."""

    # ----- utilities ----------------------------------------------------
    def reply(self, original: Message, **kw: Any) -> Message:
        return original.reply(sender=self.name, **kw)

    def emit(self, task: str, context: Dict[str, Any], **kw: Any) -> Message:
        msg = Message(
            sender=self.name, receiver="*", task=task,
            type=MessageType.EVENT, context=context,
            source_repo=self.repository, **kw,
        )
        self._bus.send(msg)
        return msg

    def request(self, other: str, task: str,
                context: Optional[Dict[str, Any]] = None,
                timeout: float = 2.0) -> Optional[Message]:
        msg = Message(
            sender=self.name, receiver=other, task=task,
            type=MessageType.REQUEST, context=context or {},
            source_repo=self.repository,
        )
        return self._bus.request(msg, timeout=timeout)

    # ----- introspection -----------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "repository": self.repository,
                "handled": self._tasks_handled,
                "failed": self._tasks_failed,
                "last_activity": self._last_activity_ts,
                "running": not self._stop_event.is_set(),
                "domain": self.domain,
                "memory_namespace": self.memory_namespace,
                "security_level": self.security_level,
            }

    def constitutional_profile(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repository": self.repository,
            "domain": self.domain,
            "mission": dict(self.mission or {}),
            "capabilities": list(self.capabilities),
            "memory_namespace": self.memory_namespace,
            "security_level": self.security_level,
            "lifecycle_status": self.lifecycle_status,
        }

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_sec):
            try:
                self._bus.send(heartbeat(self.name, context=self.stats()))
            except Exception as exc:
                log.debug("heartbeat failed for %s: %s", self.name, exc)
