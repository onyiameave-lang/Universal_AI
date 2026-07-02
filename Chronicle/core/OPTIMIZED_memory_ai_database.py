# memory_ai_database.py
# OPTIMIZED MEMORY AI - Database Learning, Training & Testing System
# Handles agent knowledge, trains on data, tests effectiveness

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

class MemoryDatabase:
    """
    Persistent database for Memory AI knowledge.
    Stores knowledge records, agent sources, and optimization history.
        
    NOTE:
    - The existing `concepts` table remains as-is for backward compatibility.
    - New Knowledge Domain Layer tables are added below.
    """

    def __init__(self, db_path: str = "memory_ai.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        """Create database schema for Memory AI."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Core knowledge table (legacy)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (

                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                concept TEXT NOT NULL,
                what TEXT NOT NULL,
                why TEXT NOT NULL,
                when_to_use TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(domain, concept, source_agent)
            )
        """)

        # =========================================================
        # Knowledge Domain Layer (new)
        # =========================================================
        # This is the single source of truth for migrated knowledge.
        # We store embeddings as JSON-encoded float arrays for now
        # (vector indexing will be added in a later Phase B step).

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_records (
                knowledge_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                source TEXT,
                title TEXT,
                author TEXT,
                category TEXT,
                symbol TEXT,
                strategy_type TEXT,
                market_regime TEXT,
                importance_score REAL DEFAULT 0.0,
                embedding_json TEXT,
                summary TEXT,
                relationships_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_lifecycles (
                lifecycle_id TEXT PRIMARY KEY,
                knowledge_id TEXT,
                strategy_name TEXT,
                symbol TEXT,
                creation_date TEXT,
                creator_agent TEXT,
                logic_json TEXT,
                parameters_json TEXT,
                optimization_history_json TEXT,
                performance_history_json TEXT,
                failure_history_json TEXT,
                drawdown_history_json TEXT,
                market_regimes_json TEXT,
                profitability_metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(knowledge_id) REFERENCES knowledge_records(knowledge_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_profiles (
                symbol_profile_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                volatility_patterns_json TEXT,
                market_behavior_json TEXT,
                successful_strategies_json TEXT,
                failed_strategies_json TEXT,
                regime_performance_json TEXT,
                session_statistics_json TEXT,
                historical_outcomes_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol)
            )
        """)

        # Optional: relationship edge storage (for knowledge linking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                relationship_id TEXT PRIMARY KEY,
                from_knowledge_id TEXT NOT NULL,
                to_knowledge_id TEXT NOT NULL,
                relationship_type TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(from_knowledge_id) REFERENCES knowledge_records(knowledge_id),
                FOREIGN KEY(to_knowledge_id) REFERENCES knowledge_records(knowledge_id)
            )
        """)

        # Agent contributions tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                concept_id INTEGER,
                raw_content TEXT NOT NULL,
                processing_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        """)
        
        # Training data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                concept_id INTEGER,
                training_example TEXT NOT NULL,
                expected_output TEXT,
                actual_output TEXT,
                accuracy_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        """)
        
        # Test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                concept_id INTEGER,
                test_query TEXT NOT NULL,
                expected_answer TEXT NOT NULL,
                generated_answer TEXT NOT NULL,
                accuracy_score REAL DEFAULT 0.0,
                test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        """)
        
        # Optimization history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                concept_id INTEGER,
                optimization_type TEXT NOT NULL,
                before_score REAL,
                after_score REAL,
                improvement REAL,
                optimization_details TEXT,
                optimized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
        """)
        
        self.conn.commit()
    
    def add_concept(self, domain: str, concept: str, what: str, why: str, 
                   when_to_use: str, source_agent: str, confidence: float = 0.5) -> int:
        """Add or update a concept in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO concepts 
            (domain, concept, what, why, when_to_use, source_agent, confidence_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (domain, concept, what, why, when_to_use, source_agent, confidence, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_concept(self, domain: str, concept: str) -> Optional[Dict]:
        """Retrieve a concept."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM concepts 
            WHERE domain = ? AND concept = ?
            ORDER BY effectiveness_score DESC LIMIT 1
        """, (domain, concept))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_domain_concepts(self, domain: str) -> List[Dict]:
        """Get all concepts in a domain."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM concepts 
            WHERE domain = ?
            ORDER BY effectiveness_score DESC
        """, (domain,))
        return [dict(row) for row in cursor.fetchall()]
    
    def increment_usage(self, concept_id: int):
        """Track concept usage."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE concepts 
            SET usage_count = usage_count + 1
            WHERE id = ?
        """, (concept_id,))
        self.conn.commit()
    
    def update_effectiveness(self, concept_id: int, score: float):
        """Update concept effectiveness score."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE concepts 
            SET effectiveness_score = ?, updated_at = ?
            WHERE id = ?
        """, (score, datetime.now(), concept_id))
        self.conn.commit()
    
    def add_training_data(self, domain: str, concept_id: int, example: str, 
                         expected_output: str) -> int:
        """Add training example."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO training_data 
            (domain, concept_id, training_example, expected_output)
            VALUES (?, ?, ?, ?)
        """, (domain, concept_id, example, expected_output))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_test_result(self, domain: str, concept_id: int, query: str, 
                       expected: str, generated: str, accuracy: float):
        """Record test result."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO test_results 
            (domain, concept_id, test_query, expected_answer, generated_answer, accuracy_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (domain, concept_id, query, expected, generated, accuracy))
        self.conn.commit()
    
    def get_statistics(self, domain: Optional[str] = None) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        if domain:
            cursor.execute("SELECT COUNT(*) as count FROM concepts WHERE domain = ?", (domain,))
            concept_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT AVG(effectiveness_score) as avg_score FROM concepts WHERE domain = ?", (domain,))
            avg_score = cursor.fetchone()['avg_score'] or 0.0
            
            cursor.execute("SELECT AVG(accuracy_score) as avg_accuracy FROM test_results WHERE domain = ?", (domain,))
            avg_accuracy = cursor.fetchone()['avg_accuracy'] or 0.0
        else:
            cursor.execute("SELECT COUNT(*) as count FROM concepts")
            concept_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT AVG(effectiveness_score) as avg_score FROM concepts")
            avg_score = cursor.fetchone()['avg_score'] or 0.0
            
            cursor.execute("SELECT AVG(accuracy_score) as avg_accuracy FROM test_results")
            avg_accuracy = cursor.fetchone()['avg_accuracy'] or 0.0
        
        return {
            "total_concepts": concept_count,
            "avg_effectiveness": avg_score,
            "avg_test_accuracy": avg_accuracy,
            "domain": domain or "all"
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class AgentLearningPipeline:
    """
    Pipeline for processing agent contributions into Memory AI knowledge.
    Validates, transforms, and stores agent learning.
    """
    
    def __init__(self, database: MemoryDatabase, transformer: 'DataTransformer'):
        self.database = database
        self.transformer = transformer
        self.contribution_queue = []
    
    def receive_agent_learning(self, agent_id: str, domain: str, 
                              content: str) -> Dict:
        """Receive learning from an agent."""
        contribution = {
            "agent_id": agent_id,
            "domain": domain,
            "content": content,
            "status": "received",
            "timestamp": datetime.now().isoformat()
        }
        self.contribution_queue.append(contribution)
        return {"status": "received", "queue_position": len(self.contribution_queue)}
    
    def process_contributions(self) -> List[Dict]:
        """Process all queued contributions."""
        results = []
        
        for contrib in self.contribution_queue:
            result = self._process_single(contrib)
            results.append(result)
        
        self.contribution_queue = []
        return results
    
    def _process_single(self, contrib: Dict) -> Dict:
        """Process single contribution through validation → transformation → storage."""
        agent_id = contrib["agent_id"]
        domain = contrib["domain"]
        content = contrib["content"]
        
        # Step 1: Validate
        is_valid = self.transformer.validate_content(domain, content)
        if not is_valid:
            return {
                "status": "rejected",
                "reason": "content_validation_failed",
                "agent": agent_id
            }
        
        # Step 2: Extract concepts
        concepts = self.transformer.extract_concepts(domain, content)
        
        # Step 3: Transform to 3-Ws
        stored_concepts = []
        for concept_name, context in concepts.items():
            three_ws = self.transformer.transform_to_3ws(domain, concept_name, context)
            
            # Step 4: Store in database
            concept_id = self.database.add_concept(
                domain=domain,
                concept=concept_name,
                what=three_ws.get("what", "N/A"),
                why=three_ws.get("why", "N/A"),
                when_to_use=three_ws.get("when", "N/A"),
                source_agent=agent_id,
                confidence=three_ws.get("confidence", 0.5)
            )
            stored_concepts.append(concept_name)
        
        return {
            "status": "processed",
            "agent": agent_id,
            "domain": domain,
            "concepts_stored": stored_concepts,
            "count": len(stored_concepts)
        }


class DataTransformer(ABC):
    """
    Abstract base for transforming raw agent data into structured knowledge.
    Implement for specific LLMs or data sources.
    """
    
    @abstractmethod
    def validate_content(self, domain: str, content: str) -> bool:
        """Validate if content is worth learning."""
        pass
    
    @abstractmethod
    def extract_concepts(self, domain: str, content: str) -> Dict[str, str]:
        """Extract concept names and context."""
        pass
    
    @abstractmethod
    def transform_to_3ws(self, domain: str, concept: str, context: str) -> Dict[str, str]:
        """Transform to What/Why/When structure."""
        pass


class GeminiDataTransformer(DataTransformer):
    """Transform data using Google Gemini API."""
    
    def __init__(self):
        import os
        try:
            from google import genai as google_genai
            self.client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = "gemini-2.0-flash"
        except:
            self.client = None
    
    def validate_content(self, domain: str, content: str) -> bool:
        """Use Gemini to validate content."""
        if not self.client:
            return True
        
        prompt = f"""
        Is this content valuable for the '{domain}' domain?
        Content: {content[:200]}
        
        Reply with only: yes or no
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return "yes" in response.text.lower()
        except:
            return True
    
    def extract_concepts(self, domain: str, content: str) -> Dict[str, str]:
        """Extract key concepts using Gemini."""
        if not self.client:
            return {"general": content}
        
        prompt = f"""
        Extract key concepts from this {domain} content:
        {content[:500]}
        
        Return JSON: {{"concept_name": "brief_context", ...}}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text)
        except:
            return {"general": content}
    
    def transform_to_3ws(self, domain: str, concept: str, context: str) -> Dict[str, str]:
        """Transform to 3-Ws using Gemini."""
        if not self.client:
            return {"what": concept, "why": "N/A", "when": "N/A"}
        
        prompt = f"""
        For {domain} concept '{concept}' in context: {context[:200]}
        
        Return JSON:
        {{
            "what": "clear definition",
            "why": "why it matters",
            "when": "when to use it",
            "confidence": 0.5-1.0
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text)
        except:
            return {"what": concept, "why": "N/A", "when": "N/A", "confidence": 0.5}


class MemoryAITrainer:
    """
    Train Memory AI on domain-specific data.
    Evaluates and improves knowledge quality.
    """
    
    def __init__(self, database: MemoryDatabase, transformer: DataTransformer):
        self.database = database
        self.transformer = transformer
    
    def train_on_domain(self, domain: str, training_examples: List[Tuple[str, str]]) -> Dict:
        """Train Memory AI on domain data."""
        print(f"\n🎓 Training on {domain} domain...")
        
        trained_concepts = []
        
        for example, expected_output in training_examples:
            # Extract concepts
            concepts = self.transformer.extract_concepts(domain, example)
            
            for concept_name in concepts:
                # Get or create concept
                concept = self.database.get_concept(domain, concept_name)
                
                if concept:
                    concept_id = concept['id']
                else:
                    three_ws = self.transformer.transform_to_3ws(
                        domain, concept_name, example
                    )
                    concept_id = self.database.add_concept(
                        domain, concept_name,
                        three_ws.get("what", "N/A"),
                        three_ws.get("why", "N/A"),
                        three_ws.get("when", "N/A"),
                        "training",
                        confidence=0.7
                    )
                
                # Add training data
                self.database.add_training_data(
                    domain, concept_id, example, expected_output
                )
                trained_concepts.append(concept_name)
        
        return {
            "domain": domain,
            "trained_concepts": list(set(trained_concepts)),
            "count": len(set(trained_concepts))
        }


class MemoryAITester:
    """
    Test Memory AI effectiveness on domain knowledge.
    Measures accuracy and identifies weak concepts.
    """
    
    def __init__(self, database: MemoryDatabase, transformer: DataTransformer):
        self.database = database
        self.transformer = transformer
    
    def test_domain(self, domain: str, test_queries: List[Tuple[str, str]]) -> Dict:
        """Test Memory AI on domain."""
        print(f"\n🧪 Testing {domain} domain...")
        
        results = {
            "domain": domain,
            "total_tests": len(test_queries),
            "passed": 0,
            "failed": 0,
            "accuracy": 0.0,
            "weak_concepts": []
        }
        
        for query, expected_answer in test_queries:
            # Generate answer
            generated_answer = self._generate_answer(domain, query)
            
            # Evaluate accuracy
            accuracy = self._evaluate_accuracy(expected_answer, generated_answer)
            results["passed"] += 1 if accuracy > 0.7 else 0
            results["failed"] += 1 if accuracy <= 0.7 else 0
            
            # Store result
            concept_id = self._get_concept_id(domain, query)
            if concept_id:
                self.database.add_test_result(
                    domain, concept_id, query, expected_answer, generated_answer, accuracy
                )
        
        results["accuracy"] = results["passed"] / len(test_queries) if test_queries else 0.0
        return results
    
    def _generate_answer(self, domain: str, query: str) -> str:
        """Generate answer using Gemini."""
        concepts = self.database.get_domain_concepts(domain)
        context = json.dumps(concepts[:3], default=str)
        
        if not concepts:
            return "No concepts available for this domain."
        
        prompt = f"""
        Domain: {domain}
        Available knowledge: {context}
        
        Answer query: {query}
        """
        
        try:
            import os
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except:
            return "Unable to generate answer"
    
    def _evaluate_accuracy(self, expected: str, generated: str) -> float:
        """Simple accuracy evaluation (0-1)."""
        if not expected or not generated:
            return 0.0
        
        # Count matching words
        expected_words = set(expected.lower().split())
        generated_words = set(generated.lower().split())
        
        if not expected_words:
            return 0.0
        
        matches = len(expected_words & generated_words)
        return matches / len(expected_words)
    
    def _get_concept_id(self, domain: str, query: str) -> Optional[int]:
        """Find relevant concept ID."""
        concepts = self.database.get_domain_concepts(domain)
        for concept in concepts:
            if concept['concept'].lower() in query.lower():
                return concept['id']
        return concepts[0]['id'] if concepts else None


class MemoryAIOptimizer:
    """
    Continuously optimize Memory AI knowledge.
    Identifies weak points and improves accuracy.
    """
    
    def __init__(self, database: MemoryDatabase):
        self.database = database
    
    def optimize_domain(self, domain: str) -> Dict:
        """Optimize all concepts in a domain."""
        print(f"\n🔧 Optimizing {domain}...")
        
        concepts = self.database.get_domain_concepts(domain)
        optimizations = []
        
        for concept in concepts:
            before_score = concept['effectiveness_score']
            
            # Optimization logic (can be enhanced)
            improvement = self._calculate_improvement(concept)
            after_score = before_score + improvement
            
            # Cap at 1.0
            after_score = min(1.0, after_score)
            
            # Update in database
            self.database.update_effectiveness(concept['id'], after_score)
            
            # Record optimization
            optimizations.append({
                "concept": concept['concept'],
                "before": before_score,
                "after": after_score,
                "improvement": improvement
            })
        
        return {
            "domain": domain,
            "optimizations": optimizations,
            "avg_improvement": sum(o['improvement'] for o in optimizations) / len(optimizations) if optimizations else 0.0
        }
    
    def _calculate_improvement(self, concept: Dict) -> float:
        """Calculate improvement potential."""
        usage = concept['usage_count']
        current_score = concept['effectiveness_score']
        
        # More usage + lower score = more improvement potential
        improvement = (0.5 - current_score) * 0.1 if current_score < 0.5 else 0.01
        return improvement

