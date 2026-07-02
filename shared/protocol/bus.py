"""
shared.protocol.bus
====================
A simple, process-wide message bus.

* In-process (default): thread-safe pub/sub with a small request/Reply
  table for synchronous request/response semantics.
* Redis-backed: optional, used when ecosystem components run as
  separate processes or hosts. It degrades gracefully to the
  in-process bus if redis is not installed.

The bus is transport-agnostic from the agent's perspective. An agent
only ever calls::

    bus.send(msg)
    bus.subscribe(channel, handler)
    response = bus.request(msg, timeout=2.0)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

from shared.protocol.message import (
    Message,
    MessageType,
    MessageValidationError,
)

log = logging.getLogger("ecosystem.bus")

Handler = Callable[[Message], None]


class MessageBus:
    """In-process message bus with pub/sub and request/reply."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._inbox: Deque[Message] = deque(maxlen=10000)
        self._waiters: Dict[str, threading.Event] = {}
        self._responses: Dict[str, Message] = {}
        self._dropped: int = 0
        self._sent: int = 0

    # ---- subscription --------------------------------------------------
    def subscribe(self, channel: str, handler: Handler) -> None:
        with self._lock:
            self._subs[channel].append(handler)

    def unsubscribe(self, channel: str, handler: Optional[Handler] = None) -> None:
        with self._lock:
            if handler is None:
                self._subs.pop(channel, None)
                return
            if channel in self._subs:
                self._subs[channel] = [
                    h for h in self._subs[channel] if h is not handler
                ]

    # ---- publish / send -----------------------------------------------
    def publish(self, channel: str, msg: Message) -> int:
        """Dispatch to all subscribers of channel. Returns delivery count."""
        with self._lock:
            handlers = list(self._subs.get(channel, [])) + list(
                self._subs.get("*", [])
            )
        if not handlers:
            log.debug("No subscribers for channel %s", channel)
            return 0
        delivered = 0
        for h in handlers:
            try:
                h(msg)
                delivered += 1
            except Exception as exc:  # never let a handler kill the bus
                log.exception("Handler %r failed for channel %s: %s",
                              h, channel, exc)
        return delivered

    def send(self, msg: Message) -> int:
        if msg.is_expired():
            self._dropped += 1
            log.warning("Dropping expired message %s task=%s",
                        msg.message_id, msg.task)
            return 0
        with self._lock:
            self._inbox.append(msg)
            self._sent += 1
        if msg.channel:
            return self.publish(msg.channel, msg)
        # fan out by receiver wildcard resolution
        if msg.receiver in ("*", "broadcast"):
            return self.publish("ecosystem.broadcast", msg)
        # direct send: route to a channel named after the receiver
        direct = f"agent.{msg.receiver}"
        n = self.publish(direct, msg)
        if n == 0:
            log.debug("No handler for direct channel %s", direct)
        return n

    # ---- request / response -------------------------------------------
    def request(
        self,
        msg: Message,
        timeout: float = 2.0,
        reply_channel: Optional[str] = None,
    ) -> Optional[Message]:
        """Send msg and block until a RESPONSE arrives.

        A unique reply channel is created automatically unless the
        caller provides one.
        """
        correlation = msg.message_id
        chan = reply_channel or f"_reply.{correlation}.{uuid.uuid4().hex[:6]}"
        event = threading.Event()
        with self._lock:
            self._waiters[correlation] = event

        msg.correlation_id = correlation
        msg.channel = chan
        self.send(msg)

        if not event.wait(timeout=timeout):
            with self._lock:
                self._waiters.pop(correlation, None)
            log.warning("Request %s timed out after %.2fs", correlation, timeout)
            return None
        with self._lock:
            return self._responses.pop(correlation, None)

    def deliver_reply(self, reply: Message) -> None:
        """Called by a handler that wants to satisfy a pending request."""
        cid = reply.correlation_id or reply.reply_to
        if not cid:
            return
        with self._lock:
            waiter = self._waiters.pop(cid, None)
            if waiter is None:
                return
            self._responses[cid] = reply
        waiter.set()

    # ---- inspection ----------------------------------------------------
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sent": self._sent,
                "dropped": self._dropped,
                "inbox_size": len(self._inbox),
                "subscribers": sum(len(v) for v in self._subs.values()),
            }

    def recent(self, n: int = 20) -> List[Message]:
        with self._lock:
            return list(self._inbox)[-n:]


# --------------------------------------------------------------------------
# Process-wide default bus (lazy).
# --------------------------------------------------------------------------
_default: Optional[MessageBus] = None
_default_lock = threading.Lock()


def get_bus() -> MessageBus:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = MessageBus()
    return _default


def set_bus(bus: MessageBus) -> None:
    global _default
    with _default_lock:
        _default = bus
