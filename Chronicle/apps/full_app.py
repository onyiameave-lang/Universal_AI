import time

class FullApp:
    def __init__(self, name):
        self.name = name

    def light_mode(self):
        print(f"⚡ {self.name} running in LIGHT MODE")

    def load_feed(self):
        print(f"📜 {self.name} loading FEED")

    def load_messages(self):
        print(f"💬 {self.name} loading MESSAGES")

    def load_media(self):
        print(f"🖼️ {self.name} loading MEDIA")

    def full_ready(self):
        print(f"🚀 {self.name} FULL APP ready")
