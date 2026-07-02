import random


class ReinforcementAI:
    def __init__(self):
        self.q_table = {} # state --> Action values
        self.learning_rate = 0.1
        self.discount = 0.9

    def get_state(self, memory_manager):
        return str(memory_manager.loaded_apps)

    def choose_action(self, state, actions):

        if state not in self.q_table:
           self.q_table[state] = {a : 0 for a in actions}

        # exploration vs exploitation
        if random.random() < 0.2:
           return random.choice(actions)

        return max(self.q_table[state], key=self.q_table[state].get)


    def update(self, state, action, reward, next_state):

        if state not in self.q_table:
           self.q_table[state] = {}

        if action not in self.q_table[state]:
           self.q_table[state][action] = 0

        future = max(self.q_table.get(next_state, {}).values(), default = 0)

        self.q_table[state][action] += self.learning_rate * (reward + self.discount * future - self.q_table[state][action])

