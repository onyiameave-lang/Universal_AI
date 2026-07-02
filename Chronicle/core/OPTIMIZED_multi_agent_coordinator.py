# multi_agent_knowledge_coordinator.py
# MULTI-AGENT KNOWLEDGE SHARING SYSTEM
# Enables agents to contribute, access, and learn from shared Memory AI knowledge

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class AgentRole(Enum):
    """Roles agents can have in the knowledge ecosystem."""
    LEARNER = "learner"          # Learns from knowledge base
    CONTRIBUTOR = "contributor"  # Contributes new knowledge
    VALIDATOR = "validator"       # Validates knowledge quality
    OPTIMIZER = "optimizer"       # Optimizes existing knowledge


class AgentInterface:
    """
    Standard interface for all agents to interact with Memory AI.
    Ensures consistent knowledge contribution and retrieval.
    """
    
    def __init__(self, agent_id: str, agent_name: str, domain: str, 
                 role: AgentRole, memory_ai_system: 'MemoryAISystem'):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.domain = domain
        self.role = role
        self.memory_ai = memory_ai_system
        self.knowledge_history = []
        self.last_sync = datetime.now()
    
    def contribute_knowledge(self, concept: str, what: str, why: str, 
                            when_to_use: str, confidence: float = 0.5) -> Dict:
        """Contribute knowledge to Memory AI."""
        if self.role not in [AgentRole.CONTRIBUTOR, AgentRole.OPTIMIZER]:
            return {"status": "unauthorized", "reason": "agent_role_insufficient"}
        
        contribution = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "domain": self.domain,
            "concept": concept,
            "what": what,
            "why": why,
            "when": when_to_use,
            "confidence": confidence,
            "contributed_at": datetime.now().isoformat()
        }
        
        self.knowledge_history.append(contribution)
        
        # Send to Memory AI
        result = self.memory_ai.receive_contribution(
            agent_id=self.agent_id,
            domain=self.domain,
            concept=concept,
            three_ws={"what": what, "why": why, "when": when_to_use},
            confidence=confidence
        )
        
        return result
    
    def access_knowledge(self, concept: Optional[str] = None) -> Dict:
        """Access knowledge from Memory AI."""
        if self.role == AgentRole.OPTIMIZER:
            return {"status": "unauthorized", "reason": "agent_cannot_read"}
        
        if concept:
            return self.memory_ai.get_concept(self.domain, concept)
        else:
            return self.memory_ai.get_domain_knowledge(self.domain)
    
    def propose_optimization(self, concept: str, 
                            optimization_suggestion: str) -> Dict:
        """Propose optimization for existing knowledge."""
        if self.role != AgentRole.OPTIMIZER:
            return {"status": "unauthorized", "reason": "agent_not_optimizer"}
        
        return self.memory_ai.request_optimization(
            domain=self.domain,
            concept=concept,
            suggestion=optimization_suggestion,
            optimizer_id=self.agent_id
        )
    
    def validate_knowledge(self, concept: str, validation_result: bool) -> Dict:
        """Validate knowledge quality."""
        if self.role != AgentRole.VALIDATOR:
            return {"status": "unauthorized", "reason": "agent_not_validator"}
        
        return self.memory_ai.record_validation(
            domain=self.domain,
            concept=concept,
            validator_id=self.agent_id,
            is_valid=validation_result
        )
    
    def get_agent_stats(self) -> Dict:
        """Get this agent's contribution stats."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "role": self.role.value,
            "domain": self.domain,
            "total_contributions": len(self.knowledge_history),
            "last_sync": self.last_sync.isoformat()
        }


class MultiAgentCoordinator:
    """
    Coordinates multiple agents and manages their knowledge sharing.
    Acts as broker between agents and Memory AI.
    """
    
    def __init__(self, memory_ai_system: 'MemoryAISystem'):
        self.memory_ai = memory_ai_system
        self.agents: Dict[str, AgentInterface] = {}
        self.agent_groups: Dict[str, List[str]] = {}  # Group agents by domain
        self.knowledge_requests_log: List[Dict] = []
    
    def register_agent(self, agent_id: str, agent_name: str, domain: str,
                      role: AgentRole) -> AgentInterface:
        """Register an agent with the system."""
        agent = AgentInterface(agent_id, agent_name, domain, role, self.memory_ai)
        self.agents[agent_id] = agent
        
        if domain not in self.agent_groups:
            self.agent_groups[domain] = []
        self.agent_groups[domain].append(agent_id)
        
        print(f"✓ Registered agent '{agent_name}' ({agent_id}) as {role.value} in {domain}")
        return agent
    
    def broadcast_knowledge(self, source_agent_id: str, domain: str,
                           concept: str, knowledge: Dict) -> Dict:
        """Broadcast knowledge to all agents in a domain."""
        results = {
            "source": source_agent_id,
            "domain": domain,
            "concept": concept,
            "recipients": [],
            "failures": []
        }
        
        target_agents = self.agent_groups.get(domain, [])
        
        for agent_id in target_agents:
            if agent_id == source_agent_id:
                continue
            
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    agent.access_knowledge(concept)
                    results["recipients"].append(agent_id)
                except Exception as e:
                    results["failures"].append({"agent": agent_id, "error": str(e)})
        
        return results
    
    def sync_all_agents(self) -> Dict:
        """Sync all agents with latest Memory AI knowledge."""
        sync_results = {
            "timestamp": datetime.now().isoformat(),
            "synced_agents": 0,
            "failed_agents": 0,
            "total_knowledge_synced": 0
        }
        
        for agent_id, agent in self.agents.items():
            try:
                # Pull latest knowledge
                knowledge = agent.access_knowledge()
                agent.last_sync = datetime.now()
                sync_results["synced_agents"] += 1
                
                if isinstance(knowledge, dict):
                    sync_results["total_knowledge_synced"] += len(knowledge)
            except Exception as e:
                print(f"✗ Sync failed for {agent_id}: {e}")
                sync_results["failed_agents"] += 1
        
        return sync_results
    
    def get_agent_status(self, agent_id: str) -> Dict:
        """Get detailed status of an agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "agent_not_found"}
        
        return {
            "agent_id": agent_id,
            "name": agent.agent_name,
            "role": agent.role.value,
            "domain": agent.domain,
            "contributions": len(agent.knowledge_history),
            "last_sync": agent.last_sync.isoformat(),
            "recent_contributions": agent.knowledge_history[-3:]
        }
    
    def get_ecosystem_stats(self) -> Dict:
        """Get overall ecosystem statistics."""
        stats = {
            "total_agents": len(self.agents),
            "agents_by_role": {},
            "agents_by_domain": self.agent_groups,
            "total_contributions": 0
        }
        
        for role in AgentRole:
            count = sum(1 for a in self.agents.values() if a.role == role)
            stats["agents_by_role"][role.value] = count
        
        for agent in self.agents.values():
            stats["total_contributions"] += len(agent.knowledge_history)
        
        return stats


class CrossDomainKnowledgeRelay:
    """
    Enables agents in different domains to benefit from related knowledge.
    Implements smart knowledge transfer across domains.
    """
    
    def __init__(self, memory_ai_system: 'MemoryAISystem'):
        self.memory_ai = memory_ai_system
        self.domain_connections: Dict[str, List[str]] = {}  # domain -> related domains
    
    def connect_domains(self, domain1: str, domain2: str, relevance_score: float = 0.5):
        """Establish connection between domains."""
        if domain1 not in self.domain_connections:
            self.domain_connections[domain1] = []
        if domain2 not in self.domain_connections:
            self.domain_connections[domain2] = []
        
        self.domain_connections[domain1].append(domain2)
        self.domain_connections[domain2].append(domain1)
        
        print(f"✓ Connected {domain1} ↔ {domain2} (relevance: {relevance_score})")
    
    def relay_knowledge(self, source_domain: str, concept: str, 
                       target_domain: str) -> Optional[Dict]:
        """Relay relevant knowledge between domains."""
        # Get concept from source domain
        source_concept = self.memory_ai.get_concept(source_domain, concept)
        
        if not source_concept:
            return None
        
        # Transform for target domain if needed
        adapted_concept = self._adapt_concept(
            source_concept, source_domain, target_domain
        )
        
        # Store in target domain
        return self.memory_ai.receive_contribution(
            agent_id="cross_domain_relay",
            domain=target_domain,
            concept=f"{concept}_from_{source_domain}",
            three_ws={
                "what": adapted_concept.get("what"),
                "why": adapted_concept.get("why"),
                "when": adapted_concept.get("when")
            },
            confidence=adapted_concept.get("confidence", 0.5)
        )
    
    def _adapt_concept(self, concept: Dict, from_domain: str, 
                      to_domain: str) -> Dict:
        """Adapt concept for different domain."""
        # Implementation would use LLM to adapt
        adapted = concept.copy()
        adapted["original_domain"] = from_domain
        adapted["confidence"] *= 0.8  # Reduce confidence for adapted knowledge
        return adapted
    
    def find_similar_concepts(self, domain: str, concept: str) -> Dict:
        """Find similar concepts in related domains."""
        results = {
            "source_domain": domain,
            "concept": concept,
            "similar_in_domains": {}
        }
        
        related_domains = self.domain_connections.get(domain, [])
        
        for related_domain in related_domains:
            # Find similar concepts in related domain
            concepts = self.memory_ai.get_domain_knowledge(related_domain)
            if isinstance(concepts, dict):
                similar = self._find_similar(concept, concepts)
                if similar:
                    results["similar_in_domains"][related_domain] = similar
        
        return results
    
    def _find_similar(self, concept: str, domain_concepts: Dict) -> List[str]:
        """Find similar concepts by name matching."""
        similar = []
        concept_lower = concept.lower()
        
        for name in domain_concepts.keys():
            if concept_lower in name.lower() or name.lower() in concept_lower:
                similar.append(name)
        
        return similar


class AgentPerformanceMonitor:
    """
    Monitors agent contributions and their effectiveness.
    Provides feedback for agent improvement.
    """
    
    def __init__(self):
        self.agent_metrics: Dict[str, Dict] = {}
    
    def record_contribution(self, agent_id: str, domain: str,
                           concept: str, effectiveness: float):
        """Record agent contribution effectiveness."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = {
                "total_contributions": 0,
                "avg_effectiveness": 0.0,
                "contributions_by_domain": {},
                "last_update": datetime.now().isoformat()
            }
        
        metrics = self.agent_metrics[agent_id]
        metrics["total_contributions"] += 1
        
        if domain not in metrics["contributions_by_domain"]:
            metrics["contributions_by_domain"][domain] = {
                "count": 0,
                "avg_effectiveness": 0.0
            }
        
        domain_metrics = metrics["contributions_by_domain"][domain]
        domain_metrics["count"] += 1
        
        # Update rolling average
        old_avg = metrics["avg_effectiveness"]
        total = metrics["total_contributions"]
        metrics["avg_effectiveness"] = (old_avg * (total - 1) + effectiveness) / total
        
        old_domain_avg = domain_metrics["avg_effectiveness"]
        domain_count = domain_metrics["count"]
        domain_metrics["avg_effectiveness"] = (old_domain_avg * (domain_count - 1) + effectiveness) / domain_count
        
        metrics["last_update"] = datetime.now().isoformat()
    
    def get_agent_performance(self, agent_id: str) -> Optional[Dict]:
        """Get agent performance metrics."""
        return self.agent_metrics.get(agent_id)
    
    def get_top_agents(self, domain: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Get top performing agents."""
        results = []
        
        for agent_id, metrics in self.agent_metrics.items():
            if domain:
                domain_metrics = metrics.get("contributions_by_domain", {}).get(domain)
                if domain_metrics:
                    results.append({
                        "agent_id": agent_id,
                        "effectiveness": domain_metrics["avg_effectiveness"],
                        "contributions": domain_metrics["count"]
                    })
            else:
                results.append({
                    "agent_id": agent_id,
                    "effectiveness": metrics["avg_effectiveness"],
                    "contributions": metrics["total_contributions"]
                })
        
        # Sort by effectiveness
        results.sort(key=lambda x: x["effectiveness"], reverse=True)
        return results[:limit]


class KnowledgeValidationFramework:
    """
    Framework for validating and certifying agent knowledge.
    Ensures knowledge quality before storing in Memory AI.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[callable]] = {}
        self.certified_concepts: Dict[str, set] = {}
    
    def register_domain_rules(self, domain: str, validation_functions: List[callable]):
        """Register validation rules for a domain."""
        self.validation_rules[domain] = validation_functions
        self.certified_concepts[domain] = set()
    
    def validate_concept(self, domain: str, concept: str, 
                        three_ws: Dict) -> Dict:
        """Validate concept against domain rules."""
        result = {
            "concept": concept,
            "domain": domain,
            "is_valid": True,
            "violations": []
        }
        
        rules = self.validation_rules.get(domain, [])
        
        for rule in rules:
            try:
                violation = rule(concept, three_ws)
                if violation:
                    result["is_valid"] = False
                    result["violations"].append(violation)
            except Exception as e:
                result["violations"].append(str(e))
        
        if result["is_valid"]:
            self.certified_concepts.setdefault(domain, set()).add(concept)
        
        return result
    
    def is_certified(self, domain: str, concept: str) -> bool:
        """Check if concept is certified."""
        return concept in self.certified_concepts.get(domain, set())
