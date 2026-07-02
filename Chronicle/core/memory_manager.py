from core.game_master import GameMaster
import os
import json
import time
from datetime import datetime
from core.reinforcement_ai import ReinforcementAI
from typing import Dict, Any, List, Optional

try:
    from google import genai as google_genai
    GENAI_V2 = True
except ImportError:
    import google.generativeai as google_genai
    GENAI_V2 = False

class MemoryManager:
    """
    MemoryAI: The autonomous resource management intelligence for Universal AI.
    Orchestrates RAM, agent lifecycles, and database health via predictive simulation.
    """
    def __init__(self, max_ram: int = 8192):
        self.max_ram = max_ram
        self.current_ram = 0
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.safe_zone = 100
        self.optimization_score = 0.0
        self.performance_history: List[Dict[str, Any]] = []
        self.pending_upgrades: List[tuple] = []

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.ai_client = google_genai.Client(api_key=self.api_key) if GENAI_V2 else google_genai
        if not GENAI_V2:
            google_genai.configure(api_key=self.api_key)
            
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.game_master = None
        
        # Local RL for high-speed "reflex" decisions
        self.local_rl = ReinforcementAI()
        
        # Load formulated management rules
        self.management_rules = self._load_learned_rules()

    @property
    def loaded_apps(self) -> Dict[str, Any]:
        """Legacy compatibility: returns a view of active application entities."""
        return {k: v for k, v in self.entities.items() if v.get("type") == "application" and v["status"] == "ACTIVE"}

    @property
    def current_memory(self) -> int:
        return self.current_ram

    @property
    def max_memory(self) -> int:
        return self.max_ram

    def _load_learned_rules(self) -> Dict[str, Any]:
        path = "knowledge_cache/system_master_knowledge.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("domains", {}).get("database", {}).get("concepts", {})
        return {}

    def run_decision_cycle(self, system_metrics: Dict[str, Any]):
        print(f"\n[MemoryAI] DECISION CYCLE START: {datetime.now().strftime('%H:%M:%S')}")
        
        state = self._capture_state(system_metrics)
        strategy = self._predict_and_simulate(state)
        self._execute_optimizations(strategy)
        self._evaluate_performance(strategy, state)
        
        # 1. Get possible actions for current entities
        possible_actions = [("idle", "system")]
        for eid, data in self.entities.items():
            if data['status'] == 'ACTIVE':
                possible_actions.append(("suspend", eid))
                possible_actions.append(("compress", eid))
            elif data['status'] == 'SUSPENDED':
                possible_actions.append(("warmup", eid))

        # 2. Choose action via Local RL (no API call)
        action_type, target_id = self.local_rl.choose_action(str(state), possible_actions)
        
        # 3. Execute and calculate local reward
        self._execute_local_action(action_type, target_id)
        self._evaluate_and_train_local(state, (action_type, target_id))

    def _capture_state(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "available_ram": self.max_ram - self.current_ram,
            "current_usage": self.current_ram,
            "entities_active": [k for k, v in self.entities.items() if v['status'] == 'ACTIVE'],
            "entities_suspended": [k for k, v in self.entities.items() if v['status'] == 'SUSPENDED'],
            "db_entities": [k for k, v in self.entities.items() if v['type'] == 'database'],
            "metrics": metrics,
            "timestamp": time.time(),
            "pattern_context": "morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 18 else "evening"
        }

    def _predict_and_simulate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Task: MemoryAI Optimization. State: {json.dumps(state)}
        Generate and simulate 3 optimization strategies. 
        Focus on: predictive warming of agents, database structure reorganization, and aggressive RAM compression.
        
        Return JSON:
        {{
            "prediction": {{ "next_agent": "name", "confidence": 0.0 }},
            "selected_strategy": {{
                "actions": [
                    {{ "entity": "name", "action": "suspend/warmup/archive/compress/optimize_db", "ram_change": -50 }}
                ],
                "expected_latency_reduction": "ms",
                "expected_ram_saved": "MB"
            }}
        }}
        """
        try:
            raw = self._query_gemini(prompt)
            return json.loads(self._clean_json(raw))
        except Exception:
            return {"selected_strategy": {"actions": []}}
    def _execute_local_action(self, action: str, entity_id: str):
        if action == "idle" or entity_id not in self.entities:
            return
        
        if action == "suspend":
            self._set_entity_status(entity_id, "SUSPENDED", compression=0.5)
        elif action == "warmup":
            self._set_entity_status(entity_id, "ACTIVE", compression=1.0)
        elif action == "compress":
            self._set_entity_status(entity_id, self.entities[entity_id]['status'], compression=0.7)

    def _execute_optimizations(self, strategy: Dict[str, Any]):
        actions = strategy.get("selected_strategy", {}).get("actions", [])
        for task in actions:
            entity_id = task.get("entity")
            action = task.get("action")
            if entity_id not in self.entities:
                continue
            
            if action == "suspend":
                self._set_entity_status(entity_id, "SUSPENDED", compression=0.5)
            elif action == "warmup":
                self._set_entity_status(entity_id, "ACTIVE", compression=1.0)
            elif action == "archive":
                self._set_entity_status(entity_id, "ARCHIVED", compression=0.1)
            elif action == "compress":
                self._set_entity_status(entity_id, self.entities[entity_id]['status'], compression=0.7)
            elif action == "optimize_db":
                print(f"[MemoryAI] Reorganizing DB structure for {entity_id}...")
    def _evaluate_and_train_local(self, prev_state: Dict, action: tuple):
        # Calculate reward based on system health
        score = 0.0
        if self.current_ram < self.max_ram - self.safe_zone: score += 50
        if self.current_ram > self.max_ram * 0.9: score -= 50
        
        # Cross-reference with learned Database Knowledge
        action_str = f"{action[0]} {action[1]}".lower()
        for concept, data in self.management_rules.items():
            # If our action matches a "WHEN" or "WHY" trigger in our learned PDF data
            if any(keyword in data['when'].lower() for keyword in [action[0], "memory", "usage"]):
                score += 20 # Reward for following "Best Practices"
        
        self.optimization_score = score
        next_state = self._capture_state({})
        self.local_rl.update(str(prev_state), action, score, str(next_state))
        print(f"[MemoryAI] Local RL Training: Reward {score} applied to action {action}")

    def register_entity(self, entity_id: str, entity_type: str, base_ram: int):
        if entity_id in self.entities: return
        self.entities[entity_id] = {
            "type": entity_type,
            "base_ram": base_ram,
            "current_ram": base_ram,
            "status": "ACTIVE",
            "usage_count": 1,
            "mode": "full" if base_ram > 100 else "light"
        }
        self.current_ram += base_ram
        print(f"[MemoryAI] Registered {entity_type}: {entity_id} ({base_ram}MB)")

    def _set_entity_status(self, entity_id: str, status: str, compression: float):
        entity = self.entities[entity_id]
        old_ram = entity["current_ram"]
        new_ram = int(entity["base_ram"] * compression)
        entity["status"] = status
        entity["current_ram"] = new_ram
        self.current_ram = (self.current_ram - old_ram) + new_ram
        print(f"[MemoryAI] {entity_id} -> {status}. RAM: {old_ram}MB -> {new_ram}MB")

    def _evaluate_performance(self, strategy: Dict[str, Any], prev_state: Dict[str, Any]):
        score = 0.0
        if self.current_ram < self.max_ram - self.safe_zone: score += 40
        if len(self.entities) > 0: score += 30
        self.optimization_score = score
        print(f"[MemoryAI] Cycle Score: {score}/100. System Health: {'OPTIMAL' if score > 70 else 'DEGRADED'}")

    def can_load(self, memory_size: int) -> bool:
        return self.current_ram + memory_size <= self.max_ram - self.safe_zone

    def free_memory(self, required_space: int):
        print(f"[MemoryAI] Manual trigger: Freeing {required_space}MB...")
        attempts = 0
        while not self.can_load(required_space) and attempts < 5:
            attempts += 1
            least_used = None
            least_count = float("inf")
            for eid, data in self.entities.items():
                if data['status'] != 'ACTIVE': continue
                if self.game_master and eid in getattr(self.game_master, 'recent_apps', []): continue
                
                count = data.get('usage_count', 0)
                if count < least_count:
                    least_count = count
                    least_used = eid
            
            if least_used:
                self._set_entity_status(least_used, "SUSPENDED", compression=0.3)
            else:
                break
        
        self.run_decision_cycle({"manual_trigger": "low_memory", "required": required_space})

    def _query_gemini(self, prompt: str) -> str:
        try:
            if GENAI_V2:
                response = self.ai_client.models.generate_content(model=self.model_name, contents=prompt)
                return response.text
            else:
                response = self.ai_client.generate_text(model=self.model_name, prompt=prompt)
                return getattr(response, "result", "")
        except Exception as e:
            return f"Error: {e}"

    def _clean_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        return text.strip()

    def system_status(self):
        print(f"\n📊 [MemoryAI] SYSTEM STATUS | Score: {self.optimization_score}/100")
        print(f"RAM: {self.current_ram}/{self.max_ram}MB")
        print(f"Entities: {len(self.entities)} (Active: {len([k for k,v in self.entities.items() if v['status']=='ACTIVE'])})")

    def load_app(self, app_name, memory_size, mode="light"):
        self.register_entity(app_name, "application", memory_size)

    def add_pending(self, app_name, memory_size):
        if app_name not in self.entities:
            self.pending_upgrades.append((app_name, memory_size))
            self.register_entity(app_name, "application", memory_size)
            self._set_entity_status(app_name, "SUSPENDED", 0.2)

    def try_upgrade(self):
        self.run_decision_cycle({"trigger": "upgrade_attempt"})
        for app_name, mem in self.pending_upgrades[:]:
            if self.can_load(mem):
                self.upgrade_app(app_name, mem)
                self.pending_upgrades.remove((app_name, mem))

    def upgrade_app(self, app_name, full_memory):
        if app_name in self.entities:
            entity = self.entities[app_name]
            if entity["mode"] == "full": return
            
            needed = full_memory - entity["current_ram"]
            if not self.can_load(needed): return

            self._set_entity_status(app_name, "ACTIVE", compression=1.0)
            entity["mode"] = "full"
            print(f"🚀 {app_name} upgraded to FULL.")

    def attach_game_master(self, game_master):
        self.game_master = game_master
        # MemoryAI is autonomous; it observes GameMaster metrics but does not depend on its logic
        self.metrics_source = game_master
