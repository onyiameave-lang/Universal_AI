import threading 
import time




class BackgroundWorker:
    def __init__(self, game_master, memory_manager):
        self.game_master = game_master
        self.memory_manager = memory_manager
        self.running = False


    def start(self):
        self.running = True
        thread = threading.Thread(target = self.run, daemon = True)
        thread.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
           print("\n Backgroung intelligence running")

           # Preload likely apps
           self.game_master.preload_apps()

           # try upgrading pending apps
           self.memory_manager.try_upgrade()

           # Maintain memory buffer
           self.maintain_memory()

           time.sleep(5)

    def maintain_memory(self):
        # Keep safe buffer free
        if self.memory_manager.current_memory > (self.memory_manager.max_memory - self.memory_manager.safe_zone):
           print("Background freeing memory early.....")
           self.memory_manager.free_memory(100)
