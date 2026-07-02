from core.memory_manager import MemoryManager
from core.game_master import GameMaster
from core.controller import AppController
from core.performance import PerformanceTracker
import time

# create system
memory_manager = MemoryManager()
game_master = GameMaster(memory_manager)
tracker = PerformanceTracker()
controller = AppController("Instagram", game_master, memory_manager)

# simulate user behavior
apps = ["Instagram", "YouTube", "Twitter", "Notes"]
last_prediction = None

for app in apps * 5:
    start = time.time()

    controller.open_app(app)

    end = time.time()

    tracker.track_open(start, end)

    tracker.track_memory(memory_manager)

    tracker.track_prediction(last_prediction, app)

    # update last prediction
    if game_master.master_ai.user_ai.history:
        last = game_master.master_ai.user_ai.history[-1]
        last_prediction = game_master.master_ai.user_ai.predict(last)

        time.sleep(1)

tracker.report()