from typing import Tuple
import numpy as np
from deepi.modules.initialization import Initializer


class Orthogonal(Initializer):

    def __init__(
            self,
            gain: float
    ):
        super().__init__("orthogonal")
        self.gain = gain

    def rule(self, shape: Tuple[int, ...]) -> np.ndarray:
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = np.random.randn(*flat_shape)
        if flat_shape[0] < flat_shape[1]:
            a = a.T
            q, r = np.linalg.qr(a)
            d = np.diag(r)
            ph = np.sign(d)
            q *= ph
            q = q.T
        else:
            q, r = np.linalg.qr(a)
            d = np.diag(r)
            ph = np.sign(d)
            q *= ph
        q = q.reshape(shape)
        return self.gain * q