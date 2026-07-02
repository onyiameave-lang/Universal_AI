"""
shared.protocol.registry
=========================
Directory of agents and their capabilities.

The Coordinator in UniversalAI uses this registry to:
    * discover which agents exist in the ecosystem,
    * match an incoming task to the agent best able to handle it,
    * track the load and activity of each agent,
    * detect retired / failed agents.

Agents call::

    registry.register(AgentSpec(name=..., capabilities=[...]),
                      handler=my_callable)

Once registered, the bus can route messages to the agent either by
direct addressing (receiver="agent.name") or by capability lookup.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("ecosystem.registry")


@dataclass
class AgentSpec:
    """Declarative description of an agent."""
    name: str
    capabilities: List[str] = field(default_factory=list)
    repository: str = ""
    description: str = ""
    version: str = "1.0.0"
    active: bool = True
    load: int = 0
    priority: int = 1
    accepts_channels: List[str] = field(default_factory=list)
    mission: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    memory_namespace: str = ""
    security_level: str = "standard"
    lifecycle_status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Thread-safe directory of agents."""

    def __init__(self) -> None:
        self._specs: Dict[str, AgentSpec] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        spec: AgentSpec,
        handler: Optional[Callable[..., Any]] = None,
    ) -> None:
        with self._lock:
            self._specs[spec.name] = spec
            if handler is not None:
                self._handlers[spec.name] = handler
        log.info("Registered agent %s v%s (%s)", spec.name, spec.version, spec.repository)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._specs.pop(name, None)
            self._handlers.pop(name, None)

    def get(self, name: str) -> Optional[AgentSpec]:
        with self._lock:
            return self._specs.get(name)

    def handler(self, name: str) -> Optional[Callable[..., Any]]:
        with self._lock:
            return self._handlers.get(name)

    def list(
        self,
        repository: Optional[str] = None,
        capability: Optional[str] = None,
        active_only: bool = True,
    ) -> List[AgentSpec]:
        out: List[AgentSpec] = []
        with self._lock:
            specs = list(self._specs.values())
        for s in specs:
            if active_only and not s.active:
                continue
            if repository and s.repository != repository:
                continue
            if capability and capability not in s.capabilities:
                continue
            out.append(s)
        return out

    def find_best(self, capability: str) -> Optional[AgentSpec]:
        matches = [s for s in self.list() if capability in s.capabilities]
        if not matches:
            return None
        matches.sort(key=lambda s: (-s.priority, s.load))
        return matches[0]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "agents": len(self._specs),
                "handlers": len(self._handlers),
                "by_repository": self._count_by("repository"),
                "by_capability": self._count_by_capability(),
            }

    def _count_by(self, attr: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self._specs.values():
            k = getattr(s, attr, "") or "(none)"
            out[k] = out.get(k, 0) + 1
        return out

    def _count_by_capability(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self._specs.values():
            for c in s.capabilities:
                out[c] = out.get(c, 0) + 1
        return out


_default_registry: Optional[AgentRegistry] = None
_default_registry_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    global _default_registry
    if _default_registry is None:
        with _default_registry_lock:
            if _default_registry is None:
                _default_registry = AgentRegistry()
    return _default_registry
