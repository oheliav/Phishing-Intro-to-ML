# =============================================================================
# nn.py
# Neural Network classifier (sklearn MLPClassifier):
#   - Single hidden layer (49 -> 32 -> 1)
#   - ReLU activation, sigmoid output
#   - Trained with backpropagation and gradient descent
# =============================================================================

from sklearn.neural_network import MLPClassifier

from base_model import BaseModel
from utils import computeMetrics, Timer
from config import RANDOM_SEED


class NeuralNetworkModel(BaseModel):

    def __init__(self):
        super().__init__("Neural Network")
        self.model = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            random_state=RANDOM_SEED,
            max_iter=200,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        with Timer() as t:
            self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Trained in {t}")

    def evaluate(self, X, y) -> dict:
        self._checkTrained()
        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        metrics = computeMetrics(y, y_pred, y_proba)
        metrics["model"] = self.name
        return metrics
