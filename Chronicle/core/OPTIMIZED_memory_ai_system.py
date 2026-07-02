# OPTIMIZED_memory_ai_system.py
# COMPLETE MEMORY AI SYSTEM
# Full integration of database learning, training, testing, and multi-agent knowledge sharing

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# Import optimized components
from OPTIMIZED_memory_ai_database import (
    MemoryDatabase, AgentLearningPipeline, GeminiDataTransformer,
    MemoryAITrainer, MemoryAITester, MemoryAIOptimizer
)
from OPTIMIZED_multi_agent_coordinator import (
    AgentInterface, AgentRole, MultiAgentCoordinator, 
    CrossDomainKnowledgeRelay, AgentPerformanceMonitor,
    KnowledgeValidationFramework
)


class MemoryAISystem:
    """
    Comprehensive Memory AI System - Central hub for all agent knowledge.

    Features:
    ✓ Database-backed persistent storage
    ✓ Agent learning pipelines
    ✓ Continuous training and testing
    ✓ Multi-agent knowledge coordination
    ✓ Cross-domain knowledge relay
    ✓ Performance monitoring
    ✓ Knowledge validation
    ✓ Automatic optimization

    Phase B additions:
    - Knowledge Domain Layer API surface
    - Retrieval interface stubs (embedding/vector hooks)
    - Knowledge linking interface
    """

    
    def __init__(self, db_path: str = "memory_ai.db"):
        print("\n" + "="*80)
        print("🧠 INITIALIZING OPTIMIZED MEMORY AI SYSTEM")
        print("="*80)
        
        # Core database
        print("\n[1/7] Initializing database...")
        self.database = MemoryDatabase(db_path)
        print("     ✓ Database initialized")
        
        # Data transformation
        print("[2/7] Setting up data transformer...")
        self.transformer = GeminiDataTransformer()
        print("     ✓ Data transformer ready")
        
        # Learning pipeline
        print("[3/7] Creating learning pipeline...")
        self.learning_pipeline = AgentLearningPipeline(self.database, self.transformer)
        print("     ✓ Learning pipeline active")
        
        # Training module
        print("[4/7] Setting up trainer...")
        self.trainer = MemoryAITrainer(self.database, self.transformer)
        print("     ✓ Trainer ready")
        
        # Testing module
        print("[5/7] Setting up tester...")
        self.tester = MemoryAITester(self.database, self.transformer)
        print("     ✓ Tester ready")
        
        # Optimizer
        print("[6/7] Setting up optimizer...")
        self.optimizer = MemoryAIOptimizer(self.database)
        print("     ✓ Optimizer ready")
        
        # Multi-agent coordination
        print("[7/7] Setting up multi-agent system...")
        self.agent_coordinator = MultiAgentCoordinator(self)
        self.cross_domain_relay = CrossDomainKnowledgeRelay(self)
        self.performance_monitor = AgentPerformanceMonitor()
        self.validation_framework = KnowledgeValidationFramework()
        print("     ✓ Multi-agent system ready")
        
        # Initialization tracking
        self.initialized_at = datetime.now()
        self.processing_queue = []
        
        print("\n" + "="*80)
        print("✓ MEMORY AI SYSTEM INITIALIZED AND READY")
        print("="*80)
    
    # ===========================
    # AGENT REGISTRATION
    # ===========================
    
    def register_agent(self, agent_id: str, agent_name: str, domain: str,
                      role: str = "contributor") -> AgentInterface:
        """Register an agent with Memory AI."""
        role_enum = AgentRole[role.upper()]
        return self.agent_coordinator.register_agent(
            agent_id, agent_name, domain, role_enum
        )
    
    # ===========================
    # KNOWLEDGE CONTRIBUTION
    # ===========================
    
    def receive_contribution(self, agent_id: str, domain: str, concept: str,
                            three_ws: Dict[str, str], confidence: float = 0.5) -> Dict:
        """Receive knowledge contribution from an agent."""
        
        # Validate
        validation = self.validation_framework.validate_concept(
            domain, concept, three_ws
        )
        
        if not validation["is_valid"]:
            return {
                "status": "rejected",
                "reason": "validation_failed",
                "violations": validation["violations"]
            }
        
        # Store in database
        concept_id = self.database.add_concept(
            domain=domain,
            concept=concept,
            what=three_ws.get("what", "N/A"),
            why=three_ws.get("why", "N/A"),
            when_to_use=three_ws.get("when", "N/A"),
            source_agent=agent_id,
            confidence=confidence
        )
        
        # Record performance
        self.performance_monitor.record_contribution(
            agent_id, domain, concept, confidence
        )
        
        print(f"✓ Knowledge stored: {domain}/{concept} from {agent_id}")
        
        return {
            "status": "stored",
            "concept_id": concept_id,
            "domain": domain,
            "concept": concept
        }
    
    # ===========================
    # KNOWLEDGE RETRIEVAL
    # ===========================
    
    def get_concept(self, domain: str, concept: str) -> Optional[Dict]:
        """Retrieve a specific concept."""
        concept_data = self.database.get_concept(domain, concept)
        
        if concept_data:
            concept_id = concept_data['id']
            self.database.increment_usage(concept_id)
        
        return concept_data
    
    def get_domain_knowledge(self, domain: str) -> Dict:
        """Retrieve all knowledge in a domain.

        Legacy mode: returns 3-W concept records from the `concepts` table.
        New mode: Knowledge Domain Layer will be queried via
        `search_knowledge_records()`.
        """
        concepts = self.database.get_domain_concepts(domain)
        return {
            "domain": domain,
            "concepts": concepts,
            "count": len(concepts)
        }

    # ==========================
    # Knowledge Domain Layer
    # ==========================
    def store_knowledge_record(
        self,
        knowledge_id: str,
        domain: str,
        source: str = "",
        title: str = "",
        author: str = "",
        category: str = "",
        symbol: str = "",
        strategy_type: str = "",
        market_regime: str = "",
        importance_score: float = 0.0,
        embedding: Optional[List[float]] = None,
        summary: str = "",
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Store a Knowledge Domain Layer record.

        This is a Phase B stub wired to DB schema.
        Vector indexing will be added in a later phase.
        """
        payload = {
            "knowledge_id": knowledge_id,
            "domain": domain,
            "source": source,
            "title": title,
            "author": author,
            "category": category,
            "symbol": symbol,
            "strategy_type": strategy_type,
            "market_regime": market_regime,
            "importance_score": importance_score,
            "embedding": embedding or [],
            "summary": summary,
            "relationships": relationships or [],
        }

        # DB layer currently exposes no direct method for this table.
        # We add the SQL inline here as a minimal integration point.
        cur = self.database.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO knowledge_records (
                knowledge_id, domain, source, title, author, category,
                symbol, strategy_type, market_regime, importance_score,
                embedding_json, summary, relationships_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                payload["knowledge_id"],
                payload["domain"],
                payload["source"],
                payload["title"],
                payload["author"],
                payload["category"],
                payload["symbol"],
                payload["strategy_type"],
                payload["market_regime"],
                payload["importance_score"],
                json.dumps(payload["embedding"]),
                payload["summary"],
                json.dumps(payload["relationships"]),
            ),
        )
        self.database.conn.commit()
        return {"status": "stored", "knowledge_id": knowledge_id}

    def search_knowledge_records(
        self,
        query: str,
        domain: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic retrieval stub.

        For now this performs a lightweight keyword match over summary/title
        because embeddings/vector indexing will be implemented in a later
        Phase B step.
        """
        # Lightweight semantic-ish retrieval fallback.
        # Phase B requirement: define a retrieval interface now.
        # Vector indexing will be added later; for now we combine:
        #   - exact term match in title/summary
        #   - optional overlap scoring with a simple token model
        #   - importance_score prior
        #
        # If embeddings are present in knowledge_records.embedding_json,
        # we still do not compute cosine similarity here (no vector index yet).
        cur = self.database.conn.cursor()

        # 1) Candidate selection by domain + LIKE
        if domain:
            cur.execute(
                """
                SELECT knowledge_id, domain, source, title, author, category,
                       symbol, strategy_type, market_regime, importance_score,
                       embedding_json, summary, relationships_json
                FROM knowledge_records
                WHERE domain = ? AND (title LIKE ? OR summary LIKE ?)
                ORDER BY importance_score DESC
                LIMIT 200
                """,
                (domain, f"%{query}%", f"%{query}%"),
            )
        else:
            cur.execute(
                """
                SELECT knowledge_id, domain, source, title, author, category,
                       symbol, strategy_type, market_regime, importance_score,
                       embedding_json, summary, relationships_json
                FROM knowledge_records
                WHERE title LIKE ? OR summary LIKE ?
                ORDER BY importance_score DESC
                LIMIT 200
                """,
                (f"%{query}%", f"%{query}%"),
            )

        candidates = [dict(r) for r in cur.fetchall()]
        if not candidates:
            return []

        # 2) Simple overlap scoring
        q = (query or "").lower()
        q_tokens = [t for t in re.split(r"\W+", q) if t]
        if not q_tokens:
            # No tokens => return top by importance
            candidates.sort(key=lambda x: float(x.get("importance_score") or 0.0), reverse=True)
            return candidates[:top_k]

        def score(rec: Dict[str, Any]) -> float:
            text = f"{rec.get('title','')} {rec.get('summary','')}".lower()
            # Term presence (binary)
            present = sum(1 for t in q_tokens if t in text)
            importance = float(rec.get('importance_score') or 0.0)
            # Importance prior + overlap
            return (importance * 0.7) + (present * 1.25)

        candidates.sort(key=score, reverse=True)
        return candidates[:top_k]


    def add_knowledge_relationship(
        self,
        relationship_id: str,
        from_knowledge_id: str,
        to_knowledge_id: str,
        relationship_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a relationship edge between knowledge records."""
        cur = self.database.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO knowledge_relationships (
                relationship_id, from_knowledge_id, to_knowledge_id,
                relationship_type, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                relationship_id,
                from_knowledge_id,
                to_knowledge_id,
                relationship_type,
                json.dumps(metadata or {}),
            ),
        )
        self.database.conn.commit()
        return {"status": "stored", "relationship_id": relationship_id}

    def get_relationship_neighbors(
        self,
        knowledge_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query knowledge linking graph.

        Returns neighbor edges for a knowledge_id.

        direction:
          - 'out'  : from_knowledge_id = knowledge_id
          - 'in'   : to_knowledge_id = knowledge_id
          - 'both' : either direction
        """
        cur = self.database.conn.cursor()

        where_clauses = []
        params: List[Any] = []

        if direction in {"out", "both"}:
            where_clauses.append("from_knowledge_id = ?")
            params.append(knowledge_id)
        if direction in {"in", "both"}:
            where_clauses.append("to_knowledge_id = ?")
            params.append(knowledge_id)

        # If both, combine with OR and wrap.
        if direction == "both":
            where = "(" + " OR ".join(where_clauses) + ")"
        else:
            where = where_clauses[0]

        sql = """
            SELECT relationship_id, relationship_type,
                   from_knowledge_id, to_knowledge_id,
                   metadata_json, created_at
            FROM knowledge_relationships
            WHERE """ + where + "\n"

        # Relationship type filter
        if relationship_type:
            sql += "  AND relationship_type = ?\n"
            params.append(relationship_type)

        sql += "  ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

        def _parse_meta(x: Any) -> Any:
            if x is None:
                return {}
            if isinstance(x, (dict, list)):
                return x
            try:
                return json.loads(x)
            except Exception:
                return {"raw": x}

        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append({
                "relationship_id": r["relationship_id"],
                "relationship_type": r["relationship_type"],
                "from_knowledge_id": r["from_knowledge_id"],
                "to_knowledge_id": r["to_knowledge_id"],
                "metadata": _parse_meta(r["metadata_json"]),
                "created_at": r["created_at"],
            })
        return results


    
    # ===========================
    # TRAINING
    # ===========================
    
    def train_on_domain(self, domain: str, 
                       training_data: List[Tuple[str, str]]) -> Dict:
        """Train Memory AI on domain-specific data."""
        print(f"\n📚 Training on {domain}...")
        return self.trainer.train_on_domain(domain, training_data)
    
    # ===========================
    # TESTING
    # ===========================
    
    def test_domain(self, domain: str,
                   test_queries: List[Tuple[str, str]]) -> Dict:
        """Test Memory AI on domain."""
        print(f"\n🧪 Testing {domain}...")
        return self.tester.test_domain(domain, test_queries)
    
    # ===========================
    # OPTIMIZATION
    # ===========================
    
    def optimize_domain(self, domain: str) -> Dict:
        """Optimize all knowledge in a domain."""
        return self.optimizer.optimize_domain(domain)
    
    def optimize_all_domains(self) -> Dict:
        """Optimize all domains."""
        print("\n🔧 Running system-wide optimization...")
        
        domains = set()
        concepts = self.database.get_domain_concepts("")  # Get all
        for concept in concepts:
            domains.add(concept['domain'])
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations": {}
        }
        
        for domain in domains:
            results["optimizations"][domain] = self.optimize_domain(domain)
        
        return results
    
    # ===========================
    # MULTI-AGENT COORDINATION
    # ===========================
    
    def sync_all_agents(self) -> Dict:
        """Sync all registered agents."""
        return self.agent_coordinator.sync_all_agents()
    
    def broadcast_knowledge(self, source_agent_id: str, domain: str,
                           concept: str) -> Dict:
        """Broadcast knowledge to all agents in domain."""
        concept_data = self.get_concept(domain, concept)
        return self.agent_coordinator.broadcast_knowledge(
            source_agent_id, domain, concept, concept_data
        )
    
    # ===========================
    # CROSS-DOMAIN LEARNING
    # ===========================
    
    def connect_domains(self, domain1: str, domain2: str, relevance: float = 0.5):
        """Connect two domains for knowledge sharing."""
        self.cross_domain_relay.connect_domains(domain1, domain2, relevance)
    
    def relay_knowledge_between_domains(self, source_domain: str, 
                                        concept: str, target_domain: str) -> Dict:
        """Relay knowledge between domains."""
        return self.cross_domain_relay.relay_knowledge(
            source_domain, concept, target_domain
        )
    
    # ===========================
    # MONITORING & REPORTING
    # ===========================
    
    def get_system_status(self) -> Dict:
        """Get complete system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "initialized_at": self.initialized_at.isoformat(),
            "database": self.database.get_statistics(),
            "agents": self.agent_coordinator.get_ecosystem_stats(),
            "top_agents": self.performance_monitor.get_top_agents(),
            "uptime_seconds": (datetime.now() - self.initialized_at).total_seconds()
        }
    
    def get_domain_report(self, domain: str) -> Dict:
        """Get detailed report for a domain."""
        concepts = self.database.get_domain_concepts(domain)
        
        return {
            "domain": domain,
            "total_concepts": len(concepts),
            "avg_effectiveness": sum(c['effectiveness_score'] for c in concepts) / len(concepts) if concepts else 0.0,
            "concepts": concepts,
            "statistics": self.database.get_statistics(domain)
        }
    
    def print_system_report(self):
        """Print formatted system report."""
        status = self.get_system_status()
        
        print("\n" + "="*80)
        print("📊 MEMORY AI SYSTEM REPORT")
        print("="*80)
        
        print(f"\n⏰ System Status:")
        print(f"   Initialized: {status['initialized_at']}")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        
        print(f"\n📁 Database:")
        db_stats = status['database']
        print(f"   Total Concepts: {db_stats['total_concepts']}")
        print(f"   Avg Effectiveness: {db_stats['avg_effectiveness']:.2f}")
        print(f"   Avg Test Accuracy: {db_stats['avg_test_accuracy']:.2f}")
        
        print(f"\n👥 Agents:")
        agent_stats = status['agents']
        print(f"   Total: {agent_stats['total_agents']}")
        for role, count in agent_stats['agents_by_role'].items():
            print(f"   - {role.capitalize()}: {count}")
        print(f"   Total Contributions: {agent_stats['total_contributions']}")
        
        print(f"\n🏆 Top Agents:")
        for agent in status['top_agents'][:3]:
            print(f"   - {agent['agent_id']}: {agent['effectiveness']:.2f}")
        
        print("\n" + "="*80)


class MemoryAIDemo:
    """
    Demonstration of Memory AI with multiple agents and domains.
    Shows training, testing, optimization, and multi-agent coordination.
    """
    
    def __init__(self):
        self.memory_ai = MemoryAISystem()
    
    def run_demo(self):
        """Run complete demonstration."""
        
        print("\n" + "="*80)
        print("🚀 RUNNING MEMORY AI DEMONSTRATION")
        print("="*80)
        
        # Register agents
        print("\n[STEP 1] Registering Agents...")
        trading_agent = self.memory_ai.register_agent(
            "trading_v1", "Trading Expert", "trading", "contributor"
        )
        chess_agent = self.memory_ai.register_agent(
            "chess_v1", "Chess Master", "chess", "contributor"
        )
        validator = self.memory_ai.register_agent(
            "validator_01", "Knowledge Validator", "trading", "validator"
        )
        
        # Agents contribute knowledge
        print("\n[STEP 2] Agents Contributing Knowledge...")
        trading_agent.contribute_knowledge(
            concept="RSI",
            what="Relative Strength Index measures momentum",
            why="Identifies overbought/oversold conditions",
            when_to_use="When analyzing price momentum",
            confidence=0.85
        )
        
        chess_agent.contribute_knowledge(
            concept="Opening Principles",
            what="Control center, develop pieces, king safety",
            why="Strong foundation for winning positions",
            when_to_use="In game opening phase",
            confidence=0.90
        )
        
        # Training
        print("\n[STEP 3] Training Memory AI...")
        trading_examples = [
            ("RSI above 70 signals overbought", "Prepare to sell"),
            ("RSI below 30 signals oversold", "Prepare to buy"),
        ]
        self.memory_ai.train_on_domain("trading", trading_examples)
        
        # Testing
        print("\n[STEP 4] Testing Memory AI...")
        test_queries = [
            ("What does RSI > 70 mean?", "Overbought condition"),
            ("When to use RSI?", "For momentum analysis"),
        ]
        test_results = self.memory_ai.test_domain("trading", test_queries)
        print(f"   Test Accuracy: {test_results['accuracy']:.0%}")
        
        # Optimization
        print("\n[STEP 5] Optimizing Knowledge...")
        self.memory_ai.optimize_domain("trading")
        
        # Multi-agent sync
        print("\n[STEP 6] Syncing All Agents...")
        sync_result = self.memory_ai.sync_all_agents()
        print(f"   Synced: {sync_result['synced_agents']} agents")
        
        # Cross-domain relay
        print("\n[STEP 7] Setting up Cross-Domain Learning...")
        self.memory_ai.connect_domains("trading", "chess", 0.6)
        
        # Final report
        print("\n[STEP 8] System Report...")
        self.memory_ai.print_system_report()
        
        print("\n" + "="*80)
        print("✓ DEMONSTRATION COMPLETE")
        print("="*80)


if __name__ == "__main__":
    demo = MemoryAIDemo()
    demo.run_demo()

