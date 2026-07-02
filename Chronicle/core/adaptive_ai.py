class AdaptiveAI:
    def __init__(self):
        self.weights = {
           "frequency" : 1.0,
           "recency" : 1.0,
           "memory_cost" : -0.5
         }

    def score(self, app, history, memory_manager):
        freq = history.count(app)

        recency = 0


        if app in history:
           recency = len(history) - history[::-1].index(app)

        memory = memory_manager.loaded_apps.get(app, {}).get("memory", 0)

        score = (
           self.weights["frequency"] * freq +
           self.weights["recency"] * recency +
           self.weights["memory_cost"] * memory
        )

        return score


    def adjust(self, reward):
        # Simple learning adjeustment
        if reward > 0:
           self.weights["frequency"] += 0.01
           self.weights["recency"] += 0.01

        else:
           self.weights["memory_cost"] -= 0.01
