# database_learning_implementation.py
# Practical implementation of database learning for Memory AI

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import threading
import time

class DatabaseLearningEngine:
    """
    Practical implementation of how Memory AI learns to handle the database.
    Integrates directly with MemoryDatabase and tracks improvement over time.
    """
    
    def __init__(self, database, memory_ai_system):
        self.database = database
        self.memory_ai = memory_ai_system
        
        # Learning metrics
        self.access_log = []
        self.performance_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
        }
        
        # Learned strategies
        self.learned_strategies = {
            "cached_concepts": [],
            "archived_concepts": [],
            "indexed_concepts": [],
            "optimized_domains": [],
        }
        
        # Learning history
        self.learning_improvements = []
    
    def track_database_operation(self, operation_type: str, 
                                concept_id: int, duration_ms: float, 
                                success: bool, cache_hit: bool = False):
        """Track every database operation for learning."""
        
        self.access_log.append({
            "operation": operation_type,
            "concept_id": concept_id,
            "duration_ms": duration_ms,
            "success": success,
            "cache_hit": cache_hit,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update metrics
        self.performance_metrics["total_operations"] += 1
        if success:
            self.performance_metrics["successful_operations"] += 1
        else:
            self.performance_metrics["failed_operations"] += 1
        
        # Update running average
        total = self.performance_metrics["total_operations"]
        old_avg = self.performance_metrics["avg_response_time"]
        self.performance_metrics["avg_response_time"] = (
            (old_avg * (total - 1) + duration_ms) / total
        )
    
    def analyze_access_patterns(self) -> Dict:
        """Analyze how concepts are being accessed."""
        
        if not self.access_log:
            return {}
        
        analysis = {
            "most_accessed_concepts": {},
            "slowest_operations": [],
            "error_patterns": [],
            "cache_opportunities": [],
        }
        
        # Count accesses per concept
        concept_access_count = {}
        for entry in self.access_log:
            concept_id = entry["concept_id"]
            concept_access_count[concept_id] = concept_access_count.get(concept_id, 0) + 1
        
        # Get most accessed
        sorted_concepts = sorted(
            concept_access_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        analysis["most_accessed_concepts"] = sorted_concepts[:10]
        
        # Find slow operations
        slow_ops = sorted(
            self.access_log,
            key=lambda x: x["duration_ms"],
            reverse=True
        )[:5]
        analysis["slowest_operations"] = slow_ops
        
        # Analyze failures
        failures = [log for log in self.access_log if not log["success"]]
        if failures:
            analysis["error_patterns"].append({
                "error_rate": len(failures) / len(self.access_log),
                "recent_failures": failures[-5:]
            })
        
        # Cache opportunities (frequently accessed without hits)
        non_cached_frequent = [
            concept_id for concept_id, count in sorted_concepts
            if count > 50 and concept_id not in self.learned_strategies["cached_concepts"]
        ]
        if non_cached_frequent:
            analysis["cache_opportunities"] = non_cached_frequent[:10]
        
        return analysis
    
    def learn_optimal_configuration(self) -> Dict:
        """Memory AI learns the optimal database configuration."""
        
        analysis = self.analyze_access_patterns()
        
        learning_decisions = {
            "caching_decisions": [],
            "indexing_decisions": [],
            "archiving_decisions": [],
            "schema_suggestions": [],
        }
        
        # Decision 1: What to cache?
        if analysis.get("cache_opportunities"):
            for concept_id in analysis["cache_opportunities"]:
                learning_decisions["caching_decisions"].append({
                    "concept_id": concept_id,
                    "action": "cache_in_memory",
                    "reason": "High frequency + no cache hits",
                    "expected_improvement": "30-50% faster access"
                })
                self.learned_strategies["cached_concepts"].append(concept_id)
        
        # Decision 2: What to index?
        most_accessed = [c[0] for c in analysis.get("most_accessed_concepts", [])[:5]]
        for concept_id in most_accessed:
            if concept_id not in self.learned_strategies["indexed_concepts"]:
                learning_decisions["indexing_decisions"].append({
                    "concept_id": concept_id,
                    "action": "create_index",
                    "reason": "Frequently accessed",
                    "expected_improvement": "Faster queries"
                })
                self.learned_strategies["indexed_concepts"].append(concept_id)
        
        # Decision 3: What to archive?
        concepts = self.database.get_all_concepts()
        for concept in concepts:
            if (concept['usage_count'] < 5 and 
                concept['id'] not in self.learned_strategies["archived_concepts"]):
                learning_decisions["archiving_decisions"].append({
                    "concept_id": concept['id'],
                    "action": "move_to_archive",
                    "reason": f"Low usage ({concept['usage_count']} times)",
                    "expected_improvement": "Smaller database"
                })
                self.learned_strategies["archived_concepts"].append(concept['id'])
        
        # Decision 4: Schema suggestions
        if self.performance_metrics["avg_response_time"] > 100:
            learning_decisions["schema_suggestions"].append({
                "suggestion": "Add query indexes",
                "current_avg_response": f"{self.performance_metrics['avg_response_time']:.2f}ms",
                "target_avg_response": "50ms"
            })
        
        return learning_decisions
    
    def apply_learned_optimizations(self) -> Dict:
        """Apply optimizations that Memory AI has learned."""
        
        decisions = self.learn_optimal_configuration()
        applied = {
            "caching_applied": 0,
            "indexing_applied": 0,
            "archiving_applied": 0,
            "improvements": []
        }
        
        # Apply caching
        for decision in decisions["caching_decisions"]:
            concept_id = decision["concept_id"]
            concept = self.database.get_concept_by_id(concept_id)
            
            # Store in cache (simplified - in real impl use Redis or memcached)
            self._cache_concept(concept)
            applied["caching_applied"] += 1
            applied["improvements"].append(
                f"Cached {concept['concept']}"
            )
        
        # Apply indexing
        for decision in decisions["indexing_decisions"]:
            concept_id = decision["concept_id"]
            concept = self.database.get_concept_by_id(concept_id)
            
            # Create database index
            self._create_concept_index(concept['concept'])
            applied["indexing_applied"] += 1
            applied["improvements"].append(
                f"Indexed {concept['concept']}"
            )
        
        # Apply archiving
        for decision in decisions["archiving_decisions"]:
            concept_id = decision["concept_id"]
            concept = self.database.get_concept_by_id(concept_id)
            
            # Move to archive
            self._archive_concept(concept)
            applied["archiving_applied"] += 1
            applied["improvements"].append(
                f"Archived {concept['concept']}"
            )
        
        # Record improvement
        improvement_record = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": applied,
            "before_avg_response": self.performance_metrics["avg_response_time"],
            "after_avg_response": None,  # Measured after
        }
        self.learning_improvements.append(improvement_record)
        
        return applied
    
    def _cache_concept(self, concept: Dict):
        """Cache a concept for faster access."""
        # In production, use Redis or similar
        print(f"  💾 Caching: {concept['concept']}")
    
    def _create_concept_index(self, concept_name: str):
        """Create database index on concept."""
        cursor = self.database.conn.cursor()
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_concept_{concept_name}
                ON concepts(concept)
                WHERE concept = ?
            """, (concept_name,))
            self.database.conn.commit()
            print(f"  🔍 Indexed: {concept_name}")
        except Exception as e:
            print(f"  ✗ Indexing failed: {e}")
        query = f"""
            CREATE INDEX IF NOT EXISTS idx_concept_{concept_name}
            ON concepts(concept)
            WHERE concept = ?
        """
        # Use the new thread-safe method in MemoryDatabase
        self.database.execute_schema_change(query, (concept_name,))
        print(f"  🔍 Indexed: {concept_name}")
    
    def _archive_concept(self, concept: Dict):
        """Move concept to archive table."""
        # In production, create archive table and move data
        print(f"  📦 Archived: {concept['concept']}")
    
    def learning_report(self) -> Dict:
        """Generate report of what Memory AI has learned."""
        
        return {
            "performance_metrics": self.performance_metrics,
            "learned_strategies": self.learned_strategies,
            "optimizations_applied": len(self.learning_improvements),
            "improvements_made": [
                imp["optimizations_applied"] 
                for imp in self.learning_improvements
            ],
            "concepts_cached": len(self.learned_strategies["cached_concepts"]),
            "concepts_indexed": len(self.learned_strategies["indexed_concepts"]),
            "concepts_archived": len(self.learned_strategies["archived_concepts"]),
        }


class ContinuousLearningThread:
    """Background thread for continuous database learning."""
    
    def __init__(self, learning_engine: DatabaseLearningEngine, 
                 interval_seconds: int = 300):
        self.learning_engine = learning_engine
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
    
    def start(self):
        """Start continuous learning."""
        self.running = True
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()
        print("🚀 Continuous learning started")
    
    def stop(self):
        """Stop continuous learning."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("⏸️  Continuous learning stopped")
    
    def _learning_loop(self):
        """Main learning loop."""
        while self.running:
            try:
                print("\n" + "="*80)
                print("🧠 MEMORY AI LEARNING CYCLE")
                print("="*80)
                
                # Analyze patterns
                print("\n[1/3] Analyzing database access patterns...")
                patterns = self.learning_engine.analyze_access_patterns()
                
                if patterns:
                    print(f"  Most accessed: {len(patterns.get('most_accessed_concepts', []))} concepts")
                    print(f"  Cache opportunities: {len(patterns.get('cache_opportunities', []))}")
                    print(f"  Slow operations: {len(patterns.get('slowest_operations', []))}")
                
                # Learn optimal configuration
                print("\n[2/3] Learning optimal configuration...")
                decisions = self.learning_engine.learn_optimal_configuration()
                
                print(f"  Caching decisions: {len(decisions['caching_decisions'])}")
                print(f"  Indexing decisions: {len(decisions['indexing_decisions'])}")
                print(f"  Archiving decisions: {len(decisions['archiving_decisions'])}")
                
                # Apply optimizations
                print("\n[3/3] Applying optimizations...")
                applied = self.learning_engine.apply_learned_optimizations()
                
                print(f"  Applied caching: {applied['caching_applied']}")
                print(f"  Applied indexing: {applied['indexing_applied']}")
                print(f"  Applied archiving: {applied['archiving_applied']}")
                
                # Report
                report = self.learning_engine.learning_report()
                print(f"\n📊 Learning Report:")
                print(f"  Total operations: {report['performance_metrics']['total_operations']}")
                print(f"  Avg response time: {report['performance_metrics']['avg_response_time']:.2f}ms")
                print(f"  Concepts cached: {report['concepts_cached']}")
                print(f"  Concepts indexed: {report['concepts_indexed']}")
                
                print("\n" + "="*80)
                print("✓ Learning cycle complete")
                print("="*80)
                
                # Wait for next cycle
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                print(f"✗ Learning error: {e}")
                time.sleep(self.interval_seconds)


# Integration example
if __name__ == "__main__":
    from OPTIMIZED_memory_ai_system import MemoryAISystem
    
    # Initialize
    memory_ai = MemoryAISystem()
    
    # Create learning engine
    learner = DatabaseLearningEngine(memory_ai.database, memory_ai)
    
    # Start continuous learning
    learning_thread = ContinuousLearningThread(learner, interval_seconds=60)
    learning_thread.start()
    
    # Register and use
    agent = memory_ai.register_agent("trader_01", "Trading Bot", "trading", "contributor")
    
    # Contribute knowledge
    agent.contribute_knowledge(
        concept="RSI",
        what="Relative Strength Index",
        why="Measures momentum",
        when_to_use="For overbought/oversold",
        confidence=0.85
    )
    
    # Use it many times (Memory AI learns it's important)
    for i in range(100):
        start = time.time()
        concept = memory_ai.get_concept("trading", "RSI")
        duration = (time.time() - start) * 1000
        
        # Track operation
        learner.track_database_operation(
            operation_type="get_concept",
            concept_id=concept['id'] if concept else None,
            duration_ms=duration,
            success=concept is not None
        )
    
    # Train and test
    memory_ai.train_on_domain("trading", [
        ("RSI > 70", "Overbought"),
        ("RSI < 30", "Oversold"),
    ])
    
    results = memory_ai.test_domain("trading", [
        ("What is RSI?", "Relative Strength Index"),
    ])
    
    # Let learning happen for a bit
    time.sleep(5)
    
    # Get final report
    print("\n" + "="*80)
    print("📈 FINAL LEARNING REPORT")
    print("="*80)
    report = learner.learning_report()
    print(json.dumps(report, indent=2))
    
    # Stop learning
    learning_thread.stop()

