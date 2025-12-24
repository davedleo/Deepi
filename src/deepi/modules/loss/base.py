from abc import abstractmethod
from typing import Optional

import numpy as np

from deepi.modules import Module


_REDUCTIONS = {None, "mean", "sum"}


class Loss(Module):

    def __init__(
        self,
        _type: str,
        reduction: Optional[str]
    ):
        super().__init__(f"loss.{_type}", False)
        if reduction not in _REDUCTIONS:
            raise ValueError()
        self.reduction = reduction

        del self.next
        self.prev: Optional[Module] = None

    @abstractmethod
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute element-wise loss."""
        pass

    @abstractmethod
    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradients of the loss."""
        pass

    def link(self, module: Module): 
        pass

    def apply_reduction(self, loss: np.ndarray) -> np.ndarray:
        """Apply reduction to loss tensor."""
        if self.reduction is None:
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        loss = self.forward(y_hat, y)
        loss = self.apply_reduction(loss)

        if self._is_training:
            self.x = y_hat, y
            self.y = loss

        return loss

    def backward(self, return_gradient: bool = True) -> np.ndarray:
        y_hat, y = self.x
        gradient = self.gradients(y_hat, y)
        if self.reduction and self.reduction == "mean":
            gradient /= gradient.shape[0]

        self.dy = gradient

        if self.prev is not None: 
            self.prev.backward(gradient)

        if return_gradient:
            return gradient