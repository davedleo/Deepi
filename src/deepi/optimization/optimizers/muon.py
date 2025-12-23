from typing import Dict, Optional, Tuple
import numpy as np

from deepi import Model
from deepi.optimization.optimizers import Optimizer
from deepi.optimization.regularization import Regularizer


class Muon(Optimizer):

    def __init__(
            self,
            model: Model,
            lr: float = 1e-3,
            weight_decay: float = 0.1,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
            ns_steps: int = 5,
            eps: float = 1e-7,
            decoupled_regularization: bool = True,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ):
        super().__init__(model, lr, regularizer, decoupled_regularization, maximize, "muon")

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_coefficients = ns_coefficients
        self.ns_steps = ns_steps
        self.eps = eps

        for module_id, module in self.modules.items():
            for k, buffer in self.buffer[module_id].items():
                buffer["momentum"] = np.zeros_like(module.params[k])
                buffer["lr"] = lr + 0.0 # Safety copy

    def _newton_schulz(self, X: np.ndarray) -> np.ndarray:
        a, b, c = self.ns_coefficients

        frob = np.linalg.norm(X, ord="fro")
        X = X / (frob + self.eps)

        G = X.T @ X
        Y = G
        Z = np.eye(G.shape[0], dtype=G.dtype)

        for _ in range(self.ns_steps):
            T = a * np.eye(G.shape[0]) + b * Y + c * (Y @ Y)
            Y = Y @ T
            Z = T @ Z

        return X @ Z

    def direction(
            self,
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray:

        B = buffer["momentum"]
        B *= self.momentum
        B += dw
        buffer["momentum"] = B

        if self.nesterov:
            B_tilde = dw + self.momentum * B
        else:
            B_tilde = B

        if B_tilde.ndim == 1:
            norm = np.linalg.norm(B_tilde)
            if norm > self.eps:
                O = B_tilde / norm
            else:
                O = B_tilde

        elif B_tilde.ndim == 2:
            O = self._newton_schulz(B_tilde)

        else:
            raise ValueError("Muon only supports 1D or 2D parameters")

        lr = buffer["lr"]

        if B_tilde.ndim == 2:
            A, Bdim = B_tilde.shape
            lr = 0.2 * lr * max(A, Bdim)
            buffer["lr"] = lr

        return lr * O