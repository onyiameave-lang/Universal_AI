import threading
import time
from app_code.app_code import AppCode
from apps.full_app import FullApp


class AppController:
    def __init__(self, name, game_master, memory_manager):
       self.name = name
       self.preview = AppCode(name)
       self.full_app = FullApp(name)

       # shared instances
       self.game_master = game_master
       self.memory_manager = memory_manager

    def open_app(self, app_name):
       # Record usage
       self.game_master.record_app(app_name)
       # Show preview of the app instantly
       self.preview.launch_preview()
       time.sleep(0.2)

       # Check memory for space before fully loading
       full_memory = getattr(self, "memory_size", 300)
       light_memory = 50

       # Decision
       if self.memory_manager.can_load(full_memory):
          self.memory_manager.load_app(self.name, full_memory)

          # Load full app in background
          thread = threading.Thread(target=self.intelligent_upgrade)
          thread.start()

       elif self.memory_manager.can_load(light_memory):
          print("low memory --> light memory")

          self.full_app.light_mode()

          # track for upgrade
          self.memory_manager.add_pending(self.name, full_memory)

          # free memory in the background
          thread = threading.Thread(
              target = self.memory_manager.free_memory, args = (full_memory,)
          ).start()

       else:
          print("Not enough memory for even light mode")
          time.sleep(0.2)

       self.game_master.preload_apps()

    def intelligent_upgrade(self):

       features = self.game_master.feature_usage.get(self.name, {})

       if not features:
          order = ["feed", "media", "messages"]
       else:
          order = sorted(features, key=features.get, reverse=True)

       for feature in order:

          if feature == "feed":
             self.full_app.load_feed()

          elif feature == "messages":
             self.full_app.load_messages()

          elif feature == "media":
             self.full_app.load_media()

          time.sleep(0.2)

       self.full_app.full_ready()