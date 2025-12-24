from typing import Optional, Dict
import numpy as np
from deepi.modules.loss import Loss

class CrossEntropy(Loss):

    def __init__(
        self,
        weights: Optional[Dict[int, float]] = None,
        reduction: Optional[str] = "mean"
    ):
        super().__init__("cross_entropy", reduction)
        self.has_weights = weights is not None

        if self.has_weights:
            labels_list = list(weights.keys())
            labels_list.sort()
            weights_list = [weights[label] for label in labels_list]
            self.labels = np.array(labels_list, dtype=int)
            self.weights = np.array(weights_list, dtype=np.float64)
            self.weights /= self.weights.sum()

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = y_hat.shape[0]
        shifted = y_hat - y_hat.max(axis=1, keepdims=True)
        exp_sum = np.exp(shifted).sum(axis=1)
        loss = -shifted[np.arange(n_samples), y] + np.log(exp_sum)

        if self.has_weights:
            loss *= self.weights[y]

        return loss

    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = y_hat.shape[0]
        shifted = y_hat - y_hat.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(n_samples), y] -= 1.0

        if self.has_weights:
            probs *= self.weights[y][:, None]

        return probs