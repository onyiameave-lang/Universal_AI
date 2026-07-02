from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

class UserAI:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.model = DecisionTreeClassifier()
        self.history = []
        self.trained = False

    def observe(self, app):
        self.history.append(app)

    def train(self):
        if len(self.history) < 3:
            return

        X = self.history[:-1]
        y = self.history[1:]

        
        self.encoder.fit(X + y)
        X_enc = self.encoder.transform(X).reshape(-1, 1)
        y_enc = self.encoder.transform(y)

        self.model.fit(X_enc, y_enc)
        self.trained = True

    def predict(self, current_app):
        if not self.trained:
            return None

        try:
            x = self.encoder.transform([current_app]).reshape(-1, 1)
            pred = self.model.predict(x)
            return self.encoder.inverse_transform(pred)[0]
        except:
            return None

    def predict_with_confidence(self, current_app):
        if not self.trained:
            return None, 0

        try:
            x = self.encoder.transform([current_app]).reshape(-1, 1)
            probs = self.model.predict_proba(x)[0]

            best_index = probs.argmax()
            confidence = probs[best_index]

            prediction = self.encoder.inverse_transform([best_index])[0]

            return prediction, confidence

        except:
            return None, 0
