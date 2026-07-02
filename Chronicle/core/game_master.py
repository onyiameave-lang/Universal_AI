from collections import Counter
import time
import random
from core.master_ai import MasterAI

class GameMaster:
    def __init__(self, memory_manager):
       # App tracking
       self.app_history = []
       self.usage = Counter()
       self.transitions = {}
       self.last_app = None
       self.memory_manager = memory_manager
       # feature tracking
       self.feature_usage = {}

       # Behavior control
       self.master_ai = MasterAI()
       self.last_preloaded = None
       self.total_events = 0
       self.recent_apps = []
       self.last_preload_time = 0
       self.cooldown = 5

    def record_agent_usage(self, domain_name):
        print(f"\n👤 System utilizing Agent: {domain_name}")

        # Update history for transitions
        if self.last_app:
            if self.last_app not in self.transitions:
                self.transitions[self.last_app] = {}
            self.transitions[self.last_app][domain_name] = self.transitions[self.last_app].get(domain_name, 0) + 1
        
        self.last_app = domain_name
        self.total_events += 1

        # update AI with agent behavior
        self.master_ai.update(domain_name)

        # let AI control system
        self.master_ai.act(self.memory_manager)

    def record_feature(self, app_name, feature):
       if app_name not in self.feature_usage:
          self.feature_usage[app_name] = {}

       self.feature_usage[app_name][feature] = self.feature_usage[app_name].get(feature, 0) +1

    def preload_apps(self):

        if time.time() - self.last_preload_time < self.cooldown:
            return

        self.last_preload_time = time.time()
        if self.total_events < 5:
           print("⏳ Learning user behavior...")
           return

        predicted = self.predict_next_app()

        if predicted:
           print(f"\n🧠 AI Predicts Next Needed Agent: {predicted}")
           # Trigger light-load of the agent via memory manager
           self.memory_manager.load_app(predicted, 100) 

           self.last_preloaded = predicted
           return

        print("⏳ Insufficient data for agent prediction.")


    def predict_next_app(self):
       if not self.last_app:
          return None

       next_apps = self.transitions.get(self.last_app, {})

       if not next_apps:
          return None

       total = sum(next_apps.values())
       best_app = max(next_apps, key=next_apps.get)
       confidence = next_apps[best_app] / total

       if confidence < 0.6:
          return None

       return best_app
