from core.user_ai import UserAI
from core.external_ai import ExternalAI
from core.reinforcement_ai import ReinforcementAI
from core.adaptive_ai import AdaptiveAI


class MasterAI:
    def __init__(self):


        self.user_ai = UserAI()
        self.external_ai = ExternalAI()
        self.rl = ReinforcementAI()
        self.adaptive = AdaptiveAI()
        self.last_prediction = None
        self.last_action_time = 0

    def update(self, entity_id):
        self.user_ai.observe(entity_id)
        self.user_ai.train()
        self.external_ai.fetch_knowledge()

    def decide(self, memory_manager):
        current = None
        if self.user_ai.history:
           current = self.user_ai.history[-1]

        predicted, confidence = self.user_ai.predict_with_confidence(current)
        self.last_prediction = predicted

        state = self.rl.get_state(memory_manager)

        possible_actions = []

        possible_actions.append(("idle", None))

        # possible decisions

        if predicted and confidence > 0.6:
           possible_actions.append(("preload", predicted))

        for entity_id in memory_manager.entities:
           possible_actions.append(("upgrade", entity_id))
           possible_actions.append(("evict", entity_id))

        if not possible_actions:
           return None

        action = self.rl.choose_action(state, possible_actions)
        return action

    def act(self, memory_manager):
        import time
        now = time.time()

        if now - self.last_action_time < 1:
           return # wait 1 second between actions

        self.last_action_time = now
        action = self.decide(memory_manager)

        predicted = self.last_prediction

        if not action:
           return

        action_type, entity_id = action

        prev_state = self.rl.get_state(memory_manager)

        # EXECUTION ACTION
        if action_type == "idle":
           pass

        if action_type == "preload":
           if entity_id in memory_manager.entities and memory_manager.entities[entity_id]['status'] == 'ACTIVE':
              print(f"[MemoryAI] {entity_id} already active.")
              return
           memory_manager.load_app(entity_id, 150) # Preloading agent weight

        elif action_type == "upgrade":
           memory_manager.upgrade_app(entity_id, 500) # Full agent activation

        elif action_type == "evict":
           if entity_id in memory_manager.entities:
              # dont evict recently used apps
              usage = self.user_ai.history[-3:] if len(self.user_ai.history) >= 3 else []

              if entity_id in usage:
                 print(f"[AI] Avoiding evicting active agent: {entity_id}")
                 return
              
              entity_data = memory_manager.entities.get(entity_id)
              if entity_data:
                  memory_manager._set_entity_status(entity_id, "SUSPENDED", 0.1)
              print(f"[AI] Suspended {entity_id}")

        # MEASURE THE RESULT
        actual = None
        if self.user_ai.history:
            actual = self.user_ai.history[-1]

        reward = self.evaluate(memory_manager, action_type, entity_id, actual)

        next_state = self.rl.get_state(memory_manager)
        self.rl.update(prev_state, action, reward, next_state)
        self.adaptive.adjust(reward)

    def evaluate(self, memory_manager, action_type, predicted, actual):
        reward = 0

        # correct prediction
        # Correct prediction
        if predicted == actual:
           reward += 2
           
           # Bonus reward if the optimized system reports this agent is high-performing
           if hasattr(memory_manager, 'memory_ai'):
               perf = memory_manager.memory_ai.performance_monitor.get_top_agents(domain=actual)
               if perf and perf[0]['effectiveness'] > 0.8:
                   reward += 1

        # wrong prediction
        # Wrong prediction
        elif predicted and predicted != actual:
           reward -= 2
           
           # Penalty if we preloaded a 'noisy' or low-priority category
           if hasattr(memory_manager, 'memory_ai'):
               concept_data = memory_manager.memory_ai.database.get_concept(actual or "general", predicted)
               if concept_data and concept_data.get('effectiveness_score', 1.0) < 0.3:
                   reward -= 1

        # wasted preload
        if action_type == "preload" and predicted != actual:
           reward -= 1

        # good memory usage
        if memory_manager.current_ram < memory_manager.max_ram:
           reward += 0.5
        else:
           reward -=1

        # being cautios is good
        if action_type == "idle" :
           reward += 0.3

        return reward
