"""
shared.config
=============
Process-wide configuration loader.

Reads environment variables (with sensible defaults) and exposes them
as a frozen dataclass. Every repository in the ecosystem imports
`get_config()` rather than reading os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _list(name: str, default: Optional[List[str]] = None,
          sep: str = ",") -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default or [])
    return [s.strip() for s in raw.split(sep) if s.strip()]


@dataclass(frozen=True)
class EcosystemConfig:
    # ---- paths ----
    ecosystem_root: str = field(default_factory=lambda: os.getenv(
        "ECOSYSTEM_ROOT", os.getcwd()))
    memory_path: str = field(default_factory=lambda: os.getenv(
        "MEMORY_PATH", "memory_store"))
    knowledge_path: str = field(default_factory=lambda: os.getenv(
        "KNOWLEDGE_PATH", "knowledge_store"))
    logs_path: str = field(default_factory=lambda: os.getenv(
        "LOGS_PATH", "logs"))
    models_path: str = field(default_factory=lambda: os.getenv(
        "MODELS_PATH", "models"))

    # ---- protocol ----
    protocol_version: str = "1.0.0"
    default_priority: int = 1
    default_ttl_ms: int = 5000

    # ---- security ----
    enable_sandbox: bool = field(default_factory=lambda: _bool("ENABLE_SANDBOX", True))
    allow_unsafe_collectors: bool = field(default_factory=lambda: _bool(
        "ALLOW_UNSAFE_COLLECTORS", False))
    require_signed_messages: bool = field(default_factory=lambda: _bool(
        "REQUIRE_SIGNED_MESSAGES", False))

    # ---- RL ----
    rl_max_positions: int = field(default_factory=lambda: _int("RL_MAX_POSITIONS", 1))
    rl_max_drawdown_pct: float = field(default_factory=lambda: _float(
        "RL_MAX_DRAWDOWN_PCT", 0.20))
    rl_max_switches_per_session: int = field(default_factory=lambda: _int(
        "RL_MAX_SWITCHES_PER_SESSION", 3))
    rl_risk_per_trade: float = field(default_factory=lambda: _float(
        "RL_RISK_PER_TRADE", 0.01))

    # ---- memory ----
    memory_embedding_dim: int = field(default_factory=lambda: _int(
        "MEMORY_EMBEDDING_DIM", 384))
    memory_max_records: int = field(default_factory=lambda: _int(
        "MEMORY_MAX_RECORDS", 1_000_000))
    memory_use_real_embeddings: bool = field(default_factory=lambda: _bool(
        "MEMORY_USE_REAL_EMBEDDINGS", False))

    # ---- coordinator ----
    coordinator_heartbeat_sec: int = field(default_factory=lambda: _int(
        "COORDINATOR_HEARTBEAT_SEC", 5))
    observer_interval_sec: int = field(default_factory=lambda: _int(
        "OBSERVER_INTERVAL_SEC", 30))
    auditor_interval_sec: int = field(default_factory=lambda: _int(
        "AUDITOR_INTERVAL_SEC", 60))

    # ---- external sources ----
    enabled_news_sources: List[str] = field(default_factory=lambda: _list(
        "ENABLED_NEWS_SOURCES",
        ["rss_feed", "newsapi", "reuters", "bloomberg"]))
    enabled_social_sources: List[str] = field(default_factory=lambda: _list(
        "ENABLED_SOCIAL_SOURCES",
        ["reddit", "x", "discord", "telegram", "forums"]))

    # ---- env passthroughs (commonly used) ----
    mt5_login: str = field(default_factory=lambda: os.getenv("MT5_LOGIN", ""))
    mt5_password: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: os.getenv("MT5_SERVER", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


_config: Optional[EcosystemConfig] = None


def get_config(reload: bool = False) -> EcosystemConfig:
    global _config
    if _config is None or reload:
        _config = EcosystemConfig()
    return _config
