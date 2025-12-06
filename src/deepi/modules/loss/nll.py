from typing import Optional, Dict
import numpy as np
from deepi.modules.loss import Loss


class NLL(Loss):

    def __init__(
        self,
        weights: Optional[Dict[int, float]] = None,
        reduction: Optional[str] = "mean"
    ):
        super().__init__("nll", reduction)
        self.has_weights = weights is not None

        if self.has_weights:
            labels_list = list(weights.keys())
            labels_list.sort()
            weights_list = [weights[label] for label in labels_list]
            self.labels = np.array(labels_list, dtype=int)
            self.weights = np.array(weights_list, dtype=np.float64)
            self.weights /= self.weights.sum()  # normalize → same as PyTorch semantics

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = y_hat.shape[0]
        loss = - y_hat[np.arange(n_samples), y]

        if self.has_weights:
            loss *= self.weights[y]

        return loss

    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = y_hat.shape[0]
        grad = np.zeros_like(y_hat)
        grad[np.arange(n_samples), y] = -1.0  # exact derivative

        if self.has_weights:
            grad *= self.weights[y][:, None]

        return grad