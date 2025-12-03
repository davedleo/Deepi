from abc import abstractmethod
import numpy as np
from typing import Dict, Optional

from deepi.modules import Module


class Loss(Module):

    def __init__(
            self,
            _type: str,
    ):
        super().__init__(f"loss.{_type}", False)

    @abstractmethod
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self) -> np.ndarray:
        raise self.dx  
    

class CrossEntropy(Loss): 

    def __init__(self, weights: Optional[Dict[int, float]] = None): 
        super().__init__("cross_entropy")
        self._has_weights = weights is not None

        if self._has_weights:
            labels_list, weights_list = [], []
            for label, weight in weights.items(): 
                labels_list.append(int(label))
                weights_list.append(float(weight))

            self.labels = np.array(labels_list, dtype=int)
            self.weights = np.array(weights, dtype=float)
            self.weights = self.weights / self.weights.sum()

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        if not self._has_weights:
            n = y.size
            loss = -np.log(probs[np.arange(n), y]).mean()

            if self._is_training:
                probs[np.arange(n), y] -= 1.0
                self.dx = probs / n

            return loss

        n = y.size
        sample_weights = self.weights[y]
        loss = -(sample_weights * np.log(probs[np.arange(n), y])).mean()

        if self._is_training:
            probs[np.arange(n), y] -= 1.0
            probs *= sample_weights[:, None]
            self.dx = probs / n

        return loss


class ElasticNet(Loss):

    def __init__(self, l1: float = 0.5, l2: float = 0.5):
        super().__init__("elasticnet")
        total = l1 + l2
        self.alpha_l1 = l1 / total
        self.alpha_l2 = l2 / total

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y
        l1_term = np.abs(diff)
        l2_term = diff ** 2

        if self._is_training:
            self.dx = (
                self.alpha_l1 * np.sign(diff) / diff.size
                + self.alpha_l2 * 2.0 * diff / diff.size
            )

        return (self.alpha_l1 * l1_term + self.alpha_l2 * l2_term).mean()


class GaussianNL(Loss):

    def __init__(self, eps: float = 1e-12):
        super().__init__("gaussian_nl")
        self.eps = eps

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y - y_hat
        var = np.var(y_hat) + self.eps

        if self._is_training:
            self.dx = diff / var / y.size

        return 0.5 * ((diff ** 2) / var + np.log(2 * np.pi * var)).mean()


class KLDiv(Loss):

    def __init__(self, eps: float = 1e-12):
        super().__init__("kld")
        self.eps = eps

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        p = np.clip(y, self.eps, 1.0)
        q = np.clip(y_hat, self.eps, 1.0)

        if self._is_training:
            self.dx = -p / q / y.size

        return (p * np.log(p / q)).sum() / y.size


class MAE(Loss):

    def __init__(self):
        super().__init__("mae")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y

        if self._is_training:
            self.dx = np.sign(diff, dtype=float) / diff.size

        return np.abs(diff).mean()
    

class ModifiedUber(Loss):

    def __init__(self, delta: float = 1.0, alpha: float = 1.5):
        super().__init__("modified_uber")
        self.delta = delta
        self.alpha = alpha

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y
        abs_diff = np.abs(diff)
        mask = abs_diff <= self.delta
        loss = np.where(
            mask,
            0.5 * diff ** 2,
            self.alpha * self.delta * (abs_diff - 0.5 * self.delta),
        )

        if self._is_training:
            self.dx = np.where(
                mask,
                diff / y.size,
                self.alpha * self.delta * np.sign(diff) / y.size,
            )

        return loss.mean()


class MSE(Loss):

    def __init__(self):
        super().__init__("mse")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y

        if self._is_training:
            self.dx = 2.0 * diff / diff.size

        return (diff ** 2).mean()


class NLL(Loss):

    def __init__(self, eps: float = 1e-12):
        super().__init__("nll")
        self.eps = eps

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        clipped = np.clip(y_hat, self.eps, 1.0)
        log_probs = np.log(clipped)

        if self._is_training:
            self.dx = -y / clipped / y.size

        return -(y * log_probs).sum() / y.size


class PoissonNL(Loss):

    def __init__(self, eps: float = 1e-12):
        super().__init__("poisson_nl")
        self.eps = eps

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        clipped = np.clip(y_hat, self.eps, None)

        if self._is_training:
            self.dx = 1 - y / clipped

        return (clipped - y * np.log(clipped)).mean()


