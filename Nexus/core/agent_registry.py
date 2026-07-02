import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def get_constitutional_agent_definitions() -> Dict[str, Dict[str, Any]]:
    """Return the constitutional roles described in the books for the ecosystem."""
    return {
        "chronicle": {
            "repository": "Chronicle",
            "domain": "memory",
            "mission": {"purpose": "Preserve knowledge", "objectives": ["store learning", "retrieve context", "support evolution"]},
            "capabilities": ["memory", "retrieval", "knowledge preservation"],
            "memory_namespace": "chronicle",
            "security_level": "high",
        },
        "oracle": {
            "repository": "Oracle",
            "domain": "trading",
            "mission": {"purpose": "Generate predictions and decisions", "objectives": ["forecast", "audit signals", "guide action"]},
            "capabilities": ["forecasting", "decision making", "risk analysis"],
            "memory_namespace": "oracle",
            "security_level": "high",
        },
        "atlas": {
            "repository": "Atlas",
            "domain": "research",
            "mission": {"purpose": "Investigate unknown problems", "objectives": ["research", "evidence gathering", "synthesis"]},
            "capabilities": ["research", "reasoning", "synthesis"],
            "memory_namespace": "atlas",
            "security_level": "standard",
        },
        "sentinel": {
            "repository": "Sentinel",
            "domain": "news",
            "mission": {"purpose": "Collect and interpret news intelligence", "objectives": ["ingest events", "assess impact", "surface signals"]},
            "capabilities": ["news intelligence", "event parsing", "signal detection"],
            "memory_namespace": "sentinel",
            "security_level": "standard",
        },
        "pulse": {
            "repository": "Pulse",
            "domain": "social",
            "mission": {"purpose": "Analyze social and community signals", "objectives": ["sentiment", "community analysis", "anomaly detection"]},
            "capabilities": ["social intelligence", "sentiment analysis", "coordination detection"],
            "memory_namespace": "pulse",
            "security_level": "standard",
        },
        "genesis": {
            "repository": "Genesis",
            "domain": "agents",
            "mission": {"purpose": "Create specialized agents", "objectives": ["agent design", "spawning", "capability growth"]},
            "capabilities": ["agent creation", "design", "bootstrapping"],
            "memory_namespace": "genesis",
            "security_level": "high",
        },
        "forge": {
            "repository": "Forge",
            "domain": "training",
            "mission": {"purpose": "Train and optimize models", "objectives": ["training", "benchmarking", "optimization"]},
            "capabilities": ["training", "optimization", "benchmarking"],
            "memory_namespace": "forge",
            "security_level": "high",
        },
        "nexus": {
            "repository": "Nexus",
            "domain": "coordination",
            "mission": {"purpose": "Coordinate the ecosystem", "objectives": ["routing", "delegation", "orchestration"]},
            "capabilities": ["coordination", "routing", "orchestration"],
            "memory_namespace": "nexus",
            "security_level": "high",
        },
        "aegis": {
            "repository": "Aegis",
            "domain": "auditing",
            "mission": {"purpose": "Govern and audit system behavior", "objectives": ["validation", "security review", "accountability"]},
            "capabilities": ["audit", "security", "governance"],
            "memory_namespace": "aegis",
            "security_level": "high",
        },
    }


class AgentRegistry:
    """
    Track and manage ALL agents (builtin + spawned).
    """

    def __init__(self, agents_dir: str = "agents", memory_ai=None):
        self.agents_dir = Path(os.path.abspath(agents_dir))
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.memory_ai = memory_ai
        self._register_builtin_agents()
        self._load_spawned_agents()

    def _register_builtin_agents(self) -> None:
        self._register_constitutional_roles()
        self._register_memory_ai()
        self._register_market_oracle()
        self._register_constitutional_scaffolds()

    def _register_constitutional_roles(self) -> None:
        for role, meta in get_constitutional_agent_definitions().items():
            self.registered_agents[role] = {
                "source": f"constitutional/{meta['repository']}",
                "type": "builtin",
                "module": None,
                "class": None,
                "instance": None,
                "metadata": meta,
            }

    def _register_memory_ai(self) -> None:
        if self.memory_ai is not None:
            self.registered_agents["memory"] = {
                "source": "MemoryAI",
                "type": "builtin",
                "module": None,
                "class": None,
                "instance": self.memory_ai,
            }

    def _register_constitutional_scaffolds(self) -> None:
        scaffold_map = {
            "atlas": ("atlas_research", "AtlasResearchAgent"),
            "forge": ("forge_training", "ForgeTrainingAgent"),
            "aegis": ("aegis_governance", "AegisGovernanceAgent"),
        }
        for role, (module_name, class_name) in scaffold_map.items():
            try:
                module = importlib.import_module(f"core.{module_name}")
                cls = getattr(module, class_name)
                self.registered_agents[role] = {
                    "source": f"constitutional/{role}",
                    "type": "builtin",
                    "module": module,
                    "class": cls,
                    "instance": cls(),
                    "metadata": get_constitutional_agent_definitions().get(role, {}),
                }
            except Exception as exc:
                self.registered_agents[role] = {
                    "source": f"constitutional/{role}",
                    "type": "builtin_unavailable",
                    "module": None,
                    "class": None,
                    "instance": None,
                    "error": str(exc),
                }

    def _register_market_oracle(self) -> None:
        market_oracle_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MarketOracle-workspace"))
        if os.path.isdir(market_oracle_root):
            if market_oracle_root not in sys.path:
                sys.path.insert(0, market_oracle_root)
            try:
                from core.market_oracle_adapter import MarketOracleAgent

                self.registered_agents["trading"] = {
                    "source": "MarketOracle",
                    "type": "builtin",
                    "module": None,
                    "class": None,
                    "instance": MarketOracleAgent(market_oracle_root, memory_ai=self.memory_ai),
                }
            except Exception as exc:
                self.registered_agents["trading"] = {
                    "source": "MarketOracle",
                    "type": "builtin_unavailable",
                    "module": None,
                    "class": None,
                    "instance": None,
                    "error": str(exc),
                }

    def _load_spawned_agents(self) -> None:
        for agent_dir in self.agents_dir.iterdir():
            if agent_dir.is_dir() and agent_dir.name.endswith("_agent"):
                domain = agent_dir.name[: -len("_agent")]
                self.register_spawned_agent(domain)

    def register_spawned_agent(self, domain: str) -> bool:
        agent_dir = self.agents_dir / f"{domain}_agent"
        if not agent_dir.exists():
            return False

        module_path = str(agent_dir)
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        try:
            agent_module = importlib.import_module(f"{domain}_agent")
            class_name = f"{domain.replace('_', ' ').replace('-', ' ').title().replace(' ', '')}Agent"
            if hasattr(agent_module, class_name):
                self.registered_agents[domain] = {
                    "source": f"spawned/{domain}",
                    "type": "spawned",
                    "module": agent_module,
                    "class": getattr(agent_module, class_name),
                    "instance": None,
                }
                return True
        except Exception:
            return False
        return False

    def get_agent(self, domain: str) -> Optional[Any]:
        if domain not in self.registered_agents:
            return None
        agent_info = self.registered_agents[domain]
        if agent_info["instance"] is not None:
            return agent_info["instance"]
        if agent_info["class"] is None:
            return agent_info.get("instance")
        try:
            instance = agent_info["class"]()
            agent_info["instance"] = instance
            return instance
        except Exception:
            return None

    def has_agent(self, domain: str) -> bool:
        return domain in self.registered_agents

    def list_agents(self) -> Dict[str, Dict[str, str]]:
        return {
            domain: {
                "type": info["type"],
                "source": info["source"],
                **({"error": info["error"]} if info.get("error") else {}),
            }
            for domain, info in self.registered_agents.items()
        }
