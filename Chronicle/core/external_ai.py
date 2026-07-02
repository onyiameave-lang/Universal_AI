import requests
import time


class ExternalAI:
    def __init__(self):
        self.knowledge = {}
        self.last_update = 0
        self.update_interval = 60 # seconds

    def fetch_knowledge(self):
        now = time.time()

        # prevent too many api calls
        if now - self.last_update < self.update_interval:
           return

        try:
           # we can replace the limk later
           response = requests.get("https://api.mocki.io/v1/ce5f60e2")

           if response.status_code == 200:
              self.knowledge = response.json()
              print("[ExternalAI] Updated knowledge from API")

        except:
           print("[ExternalAI] fetch failed")

        self.last_update = now

    def get(self, key, default=None):
        return self.knowledge.get(key, default)
