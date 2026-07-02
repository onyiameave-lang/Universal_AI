"""
shared.exceptions
==================
Hierarchy of exception types used across the entire ecosystem.
"""
from __future__ import annotations


class EcosystemError(Exception):
    """Base class."""


class ConfigurationError(EcosystemError):
    """Raised when required configuration is missing or invalid."""


class KnowledgeError(EcosystemError):
    """Raised for problems ingesting or retrieving knowledge."""


class RetrievalError(KnowledgeError):
    """Raised when a memory/knowledge lookup fails."""


class StrategyError(EcosystemError):
    """Raised when strategy creation/optimization fails."""


class RiskError(EcosystemError):
    """Raised when a risk constraint is violated."""


class ExecutionError(EcosystemError):
    """Raised when a broker/exchange call fails."""


class ValidationError(EcosystemError):
    """Raised when an evaluation step rejects a candidate strategy."""


class AgentRetiredError(EcosystemError):
    """Raised when a request is routed to a retired agent."""
