import time



class PerformanceTracker:
    def __init__(self):
        self.app_open_times = []
        self.memory_usage_log = []
        self.predictions = 0
        self.correct_predictions = 0

    def track_open(self, start, end):
        duration = end - start
        self.app_open_times.append(duration)

    def track_memory(self, memory_manager):
        self.memory_usage_log.append(memory_manager.current_memory)
        
    def track_prediction(self, predicted, actual):
        if predicted:
           self.predictions += 1
           if predicted == actual:
            self.correct_predictions +=1
    
    def report(self):
        avg_time = sum(self.app_open_times) / len(self.app_open_times) if self.app_open_times else 0
        avg_memory = sum(self.memory_usage_log) / len(self.memory_usage_log) if self.memory_usage_log else 0
        accuracy = 0
        if self.predictions > 0:
           accuracy = (self.correct_predictions/self.predictions) *1000
        print("\n📊 PERFORMANCE REPORT")
        print(f"Avg App Load Time: {avg_time:.4f} sec")
        print(f"Avg Memory Usage: {avg_memory:.2f} MB")
        print(f"Prediction Accuracy: {accuracy:.2f}%")