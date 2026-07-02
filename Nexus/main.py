import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load keys before importing modules that validate environment at import time.
env_paths = [
    ROOT_DIR / ".env",
    ROOT_DIR.parent / ".env",
]
for path in env_paths:
    if path.exists():
        load_dotenv(path)
        break

from core.agent_registry import AgentRegistry, get_constitutional_agent_definitions
from core.agent_spawner import AgentSpawnerWithSharedKB
from core.domain_classifier import DomainClassifier
from core.improvement_suggester import ImprovementSuggester
from shared.mission_manager import MissionManager

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


class UniversalAI:
    """
    Main orchestrator for the Universal AI ecosystem.
    """

    def __init__(self, memory_ai_system: Any = None):
        self.root_dir = os.path.abspath(os.path.dirname(__file__))
        self.memory_ai_system = memory_ai_system
        self.classifier = DomainClassifier()
        self.query_counts: Dict[str, int] = {}
        self.agents_dir = os.path.join(self.root_dir, "agents")
        self.agent_registry = AgentRegistry(self.agents_dir, memory_ai=memory_ai_system)
        self.agent_spawner = AgentSpawnerWithSharedKB(memory_ai_system, agents_dir=self.agents_dir)
        self.mission_manager = MissionManager()

        # Initialize and connect the AI Memory Knowledge Base
        self.knowledge_base = self._init_knowledge_base()
        self.improvement_suggester = ImprovementSuggester(
            memory_ai=memory_ai_system, 
            market_oracle=self.knowledge_base
        )
        self.started_at = time.time()

    def _init_knowledge_base(self):
        """Dynamically link the Knowledge Base script to the system."""
        try:
            kb_core_path = os.path.abspath(os.path.join(self.root_dir, "..", "ai-memory-system-", "core"))
            if kb_core_path not in sys.path:
                sys.path.insert(0, kb_core_path)
            
            from knowledge_base import MemoryAIKnowledgeBase
            kb = MemoryAIKnowledgeBase(memory_ai=self.memory_ai_system)
            kb.learn_from_sources() # Seeding initial knowledge
            return kb
        except Exception as e:
            print(f"Ecosystem Linkage Error: {e}")
            return None

    def process_query(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        classification = self.classifier.classify(query)
        domain = classification.get("domain", "general")
        confidence = float(classification.get("confidence", 0.0))
        self.query_counts[domain] = self.query_counts.get(domain, 0) + 1

        self.mission_manager.create_mission(
            mission_id=f"query:{domain}",
            description=f"Handle {domain} query",
            objectives=["understand intent", "route to best agent", "return answer"],
            domain=domain,
            metadata={"user_id": user_id},
        )

        agent_available = self.agent_registry.has_agent(domain)
        should_spawn = self.query_counts[domain] >= 5 and not agent_available
        spawn_result = None
        if should_spawn:
            spawn_result = self.agent_spawner.spawn_agent(domain)
            self._finalize_spawned_agent(domain)
            agent_available = self.agent_registry.has_agent(domain)

        response = None
        if agent_available:
            agent = self.agent_registry.get_agent(domain)
            if agent is not None:
                self.mission_manager.assign_agent(f"query:{domain}", getattr(agent, "name", domain), {"role": "handler"})
            if hasattr(agent, "answer"):
                response = agent.answer(query)
            elif domain == "trading" and self.knowledge_base:
                response = self.knowledge_base.answer_query_with_3ws(query)
            else:
                response = {"error": "Agent does not support answer()"}
        else:
            response = self._fallback_ai(query)

        if self.memory_ai_system:
            try:
                self.memory_ai_system.shared_knowledge.add_agent_source(domain, user_id, query)
            except Exception:
                pass

        return {
            "query": query,
            "domain": domain,
            "confidence": confidence,
            "agent_available": agent_available,
            "spawned_agent": bool(spawn_result),
            "spawn_result": spawn_result,
            "response": response,
            "classification": classification,
        }

    def _finalize_spawned_agent(self, domain: str) -> None:
        try:
            self.agent_registry.register_spawned_agent(domain)
            agent = self.agent_registry.get_agent(domain)
            if not agent:
                return
            for method_name in ("bootstrap", "learn", "learn_from_memory"):
                method = getattr(agent, method_name, None)
                if callable(method):
                    try:
                        method()
                    except TypeError:
                        method(self.memory_ai_system)
                    return
        except Exception:
            pass

    def _fallback_ai(self, query: str) -> Dict[str, Any]:
        prompt = f"""
You are a fallback AI agent for answering user queries.
Provide an answer with context, domain reasoning, and if possible a 3-Ws formatted response.
User Query: {query}
"""
        try:
            from universal_ai_chat_interface import UniversalAIChatInterface

            chat = UniversalAIChatInterface(
                memory_ai_system=self.memory_ai_system,
                universal_ai_system=self,
            )
            text = chat._get_ai_response([{"role": "user", "content": prompt}])
            return {"fallback": True, "answer": text}
        except Exception as exc:
            return {"fallback": True, "error": str(exc)}

    def suggest_improvements(self, domain: str) -> Dict[str, Any]:
        return self.improvement_suggester.suggest_improvements(domain)

    def optimize_all_agents(self) -> Dict[str, Any]:
        summary = {
            "optimized_agents": [],
            "errors": [],
        }
        for domain, info in self.agent_registry.list_agents().items():
            try:
                if self.memory_ai_system:
                    self.memory_ai_system.optimize_domain_strategies(domain)
                suggestion = self.improvement_suggester.suggest_improvements(domain)
                summary["optimized_agents"].append({"domain": domain, "suggestion": suggestion})
            except Exception as exc:
                summary["errors"].append({"domain": domain, "error": str(exc)})
        return summary

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "query_counts": self.query_counts,
            "agents": self.agent_registry.list_agents(),
            "memory_ai_connected": self.memory_ai_system is not None,
            "constitutional_agents": get_constitutional_agent_definitions(),
        }


def _load_memory_ai() -> Any:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-memory-system-", "core"))
    if os.path.isdir(root_dir) and root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    try:
        from OPTIMIZED_memory_ai_system import MemoryAISystem
        return MemoryAISystem()
    except Exception:
        return None


def main() -> None:
    memory_ai = _load_memory_ai()
    universal = UniversalAI(memory_ai_system=memory_ai)
    print(json.dumps(universal.get_system_status(), indent=2))
    print("Universal AI is ready. Enter queries or commands.")
    print("Commands: status | suggest <domain> | optimize | observe news_social_demo | quit")

    # Phase G orchestrator (Observer role MVP)
    phase_g_orchestrator = None
    if memory_ai is not None:
        try:
            from core.phase_g_universal_orchestrator import PhaseGUniversalOrchestrator  # type: ignore
            phase_g_orchestrator = PhaseGUniversalOrchestrator(memory_ai=memory_ai)
        except Exception:
            phase_g_orchestrator = None

    while True:
        try:
            line = input("UniversalAI> ").strip()
            if not line:
                continue
            if line.lower() in {"quit", "exit", "q"}:
                print("Exiting Universal AI.")
                break
            if line.startswith("status"):
                print(json.dumps(universal.get_system_status(), indent=2))
                continue
            if line.startswith("suggest "):
                domain = line.split(" ", 1)[1].strip()
                print(json.dumps(universal.suggest_improvements(domain), indent=2))
                continue
            if line.startswith("optimize"):
                print(json.dumps(universal.optimize_all_agents(), indent=2))
                continue

            if line.startswith("observe news_social_demo"):
                if phase_g_orchestrator is None:
                    print(json.dumps({"error": "phase_g_orchestrator_unavailable"}, indent=2))
                    continue

                # Demo intelligence payload for MVP
                intelligence: List[Dict[str, Any]] = [
                    {
                        "source": "news",
                        "event_type": "macro_economic",
                        "summary": "EURUSD affected by inflation expectations and central bank guidance",
                        "sentiment": -0.2,
                        "topics": ["EURUSD", "inflation", "ECB"],
                        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    {
                        "source": "social",
                        "event_type": "market_hype",
                        "summary": "Traders discussing BTC momentum and breakout speculation",
                        "sentiment": 0.3,
                        "topics": ["BTCUSD", "breakout", "momentum"],
                        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                ]
                payload = phase_g_orchestrator.observe_and_generate_opportunities(
                    intelligence=intelligence,
                    regime_hint="high_volatility",
                )
                print(json.dumps(payload, indent=2))
                continue

            result = universal.process_query(line)
            print(json.dumps(result, indent=2))
        except KeyboardInterrupt:
            print("\nExiting Universal AI.")
            break
        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
