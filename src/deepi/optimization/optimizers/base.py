from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from deepi import Model
from deepi.optimization.regularization import Regularizer


class Optimizer(ABC):

    def __init__(
        self,
        model: Model,
        lr: float,
        regularizer: Optional[Regularizer],
        decoupled_regularization: bool,
        maximize: bool,
        has_buffer: bool,
        _type: str,
    ):
        self.lr = lr
        self.regularizer = regularizer
        self.decoupled_regularization = decoupled_regularization
        self.sign = -1.0 if maximize else 1.0

        self.modules = {
            module_id: module
            for module_id, module in model.modules.items()
            if module.has_params
        }

        self.buffer = {
            "state": dict(),
            "params": {
                module_id: {
                    k: dict() if has_buffer else None
                    for k in module.params.keys()
                }
                for module_id, module in self.modules.items()
            }
        }

        self._type = f"optimizer.{_type}"

    @abstractmethod
    def direction(
        self,
        dw: np.ndarray,
        buffer: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        pass

    def step(self):
        for module_id, module in self.modules.items():
            params = module.params
            grads = module.grads
            module_buffer = self.buffer["params"][module_id]

            for k, w in params.items():
                # Skip running stats
                if not k.startswith("running"):
                    dw = self.sign * grads[k]
                    buffer = module_buffer[k]

                    if self.regularizer is not None:
                        if self.decoupled_regularization:
                            dp = self.direction(dw, buffer)
                            dp += self.lr * self.regularizer(w)
                        else:
                            dw_reg = dw + self.regularizer(w)
                            dp = self.direction(dw_reg, buffer)

                    else:
                        dp = self.direction(dw, buffer)

                    params[k] -= dp

    def get_buffer(self) -> Dict[str, Dict[str, Any]]:
        return self.buffer

    def load_buffer(
            self,
            buffer_dict: Dict[str, Dict[str, Any]]
    ):
        if "params" in buffer_dict:
            for module_id, buffers in buffer_dict["params"].items():
                if module_id in self.buffer["params"]:
                    for k, buf in buffers.items():
                        if k in self.buffer["params"][module_id]:
                            self.buffer["params"][module_id][k] = deepcopy(buf)

        if "state" in buffer_dict:
            self.buffer["state"] = deepcopy(buffer_dict["state"])
