"""
memory_pipeline.py — Memory AI Training Pipeline
Teaches the Memory AI how to manage databases by reading expert data.
"""

import os
from core.knowledge_base import MemoryAIKnowledgeBase
from core.strategy_optimizer import StrategyOptimizer
from core.shared_knowledge_base import SharedKnowledgeBase

def run_training_pipeline(pdf_paths: list, youtube_ids: list):
    print("\n" + "="*60)
    print("MEMORY AI: SYSTEM KNOWLEDGE PIPELINE")
    print("="*60)

    # 1. Ingestion: Learn from Database Experts
    kb = MemoryAIKnowledgeBase()
    for pdf in pdf_paths:
        print(f"Ingesting PDF: {pdf}")
        kb.learn_from_pdf(pdf, domain="database")
    
    for vid in youtube_ids:
        print(f"Ingesting Video: {vid}")
        kb.learn_from_youtube(vid, domain="database")

    # 2. Formulation: 3-Ws Structuring
    print("\nFormulating Knowledge into 3-Ws (What, Why, When)...")
    shared_kb = SharedKnowledgeBase()
    optimizer = StrategyOptimizer(shared_kb)
    
    # 3. Optimization: Clean and Refine Management Strategies
    print("\nOptimizing Database Management Strategies...")
    report = optimizer.optimize_domain_strategies("database")
    print(f"Optimization complete. Insights: {report.get('insights')}")

    # 4. Testing: Simulated Decision Cycle
    from core.memory_manager import MemoryManager
    manager = MemoryManager()
    print("\nRunning Test Decision Cycle with Learned Rules...")
    manager.run_decision_cycle({"test": True, "load": "high"})
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE: Memory AI is now an expert in Database Management")
    print("="*60)

if __name__ == "__main__":
    # Example Usage
    run_training_pipeline(
        pdf_paths=["knowledge/sql_optimization_guide.pdf"],
        youtube_ids=["dQw4w9WgXcQ"] # Replace with actual DB tutorial IDs
    )