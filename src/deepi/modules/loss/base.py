from abc import abstractmethod
from typing import Optional

import numpy as np

from deepi.modules import Module


_REDUCTIONS = {None, "mean", "sum"}


class Loss(Module):

    def __init__(
        self,
        _type: str,
        reduction: Optional[str] = "mean",
    ):
        super().__init__(f"loss.{_type}", False)
        if reduction not in _REDUCTIONS:
            raise ValueError()

        self.reduction = reduction

    @abstractmethod
    def transform(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute element-wise loss."""
        pass

    @abstractmethod
    def gradients(self) -> np.ndarray:
        """Compute gradients of the loss."""
        pass

    def apply_reduction(self, loss: np.ndarray) -> np.ndarray:
        """Apply reduction to loss tensor."""
        if self.reduction is None:
            return loss
        elif self.reduction == "sum":
            return loss.sum(keepdims=True)
        else:
            return loss.mean(keepdims=True)

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        loss = self.transform(y, y_hat)
        loss = self.apply_reduction(loss)

        if self._is_training:
            self.x = y, y_hat
            self.y = loss

        return loss

    def backward(self) -> np.ndarray:
        gradient = self.gradients()
        if self.reduction and self.reduction == "mean":
            gradient /= gradient.shape[0]

        self.dy = gradient

        for prev_module in self.prev:
            prev_module.backward(gradient)

        return gradient