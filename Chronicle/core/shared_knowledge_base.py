import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class SharedKnowledgeBase:
    """
    Central repository for all AI ecosystem knowledge.
    Coordinates storage and retrieval for multiple domains using the 3-Ws structure.
    """

    def __init__(self, storage_dir: str = "shared_memory"):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.knowledge_file = os.path.join(self.storage_dir, "shared_concepts.json")
        self.source_logs_file = os.path.join(self.storage_dir, "agent_sources.json")
        
        self.knowledge = self._load_data(self.knowledge_file, {"domains": {}})
        self.sources = self._load_data(self.source_logs_file, {"history": []})

    def _load_data(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    def _save_data(self, path: str, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_agent_source(self, domain: str, agent_id: str, content: str):
        """Logs raw data contributions from agents before processing."""
        self.sources["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain,
            "agent_id": agent_id,
            "content": content
        })
        self._save_data(self.source_logs_file, self.sources)

    def store_concept(self, domain: str, concept: str, data_3w: Dict[str, str], source: str):
        """
        Stores a processed concept with What, Why, and When attributes.
        This is the shared pool where other agents can 'grab' knowledge.
        """
        if domain not in self.knowledge["domains"]:
            self.knowledge["domains"][domain] = {"concepts": {}}
        
        self.knowledge["domains"][domain]["concepts"][concept.lower()] = {
            "concept": concept,
            "what": data_3w.get("what", "N/A"),
            "why": data_3w.get("why", "N/A"),
            "when": data_3w.get("when", "N/A"),
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save_data(self.knowledge_file, self.knowledge)

    def get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        return self.knowledge["domains"].get(domain, {"concepts": {}})

    def get_concept(self, domain: str, concept: str) -> Optional[Dict[str, Any]]:
        domain_data = self.get_domain_knowledge(domain)
        return domain_data["concepts"].get(concept.lower())

    def get_cross_domain_connections(self) -> List[Dict[str, Any]]:
        """Identifies concepts that are relevant across multiple domains."""
        connections = []
        concept_map = {}  # concept -> list of domains
        
        for domain, data in self.knowledge["domains"].items():
            for concept in data["concepts"]:
                if concept not in concept_map:
                    concept_map[concept] = []
                concept_map[concept].append(domain)
        
        for concept, domains in concept_map.items():
            if len(domains) > 1:
                connections.append({
                    "concept": concept,
                    "domains": domains
                })
        return connections


class AgentKnowledgeAccessor:
    """
    Provides a domain-specific portal for agents to access the SharedKnowledgeBase.
    """

    def __init__(self, shared_kb: SharedKnowledgeBase, domain: str):
        self.shared_kb = shared_kb
        self.domain = domain

    def get_local_knowledge(self) -> Dict[str, Any]:
        """Access knowledge specific to this agent's domain."""
        return self.shared_kb.get_domain_knowledge(self.domain)

    def browse_shared_pool(self, keyword: str) -> List[Dict[str, Any]]:
        """Allows an agent to search the entire ecosystem for related knowledge."""
        results = []
        kw = keyword.lower()
        for domain, data in self.shared_kb.knowledge["domains"].items():
            for concept, details in data["concepts"].items():
                if kw in concept or kw in details["what"].lower():
                    results.append({"domain": domain, "details": details})
        return results
    