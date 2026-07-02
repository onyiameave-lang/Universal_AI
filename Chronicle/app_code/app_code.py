class AppCode:
    def __init__(self, name):
       self.name = name

       # Simulate cache data(tiny code having to be the preview before asynchronous work)
       self.cached_feed = [
          "Post 1 (cached)"
          "Post 2 (cached)"
          "Post 3 (cached)"
       ]

    def launch_preview(self):
       print(f"Opening {self.name} instantly...\n")

       # UI skeleton
       print("="*30)
       print(f"         {self.name}")
       print("="*30, "\n")

       # Show cached content
       print(" Feed(cached)")
       for post in self.cached_feed:
          print(f"- {post}")

       # Hint that the real system is loading in the background
       print("\n Loading new content in the background\n")


