from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple

import numpy as np

from deepi.modules import Input, Module
from deepi.modules.initialization import Initializer, KaimingUniform

ArrayOrTuple = Union[np.ndarray, Tuple[np.ndarray, ...]]


class Model(ABC): 

    def __init__(
            self, 
            _type: str,
            inputs: Union[Input, List[Input]],
            outputs: Union[Module, List[Module]],
            initializer: Optional[Initializer] = None
    ): 
        self._type = f"model.{_type}"

        self.inputs: Union[Input, List[Input]] = inputs
        self.outputs: Union[Input, List[Module]] = outputs
        self.modules: List[Module] = []

        self.build(inputs, outputs)
        self._map_modules()

        if initializer is None: 
            initializer = KaimingUniform()

        self._init_params(initializer)

    @abstractmethod 
    def build(
            self,
            inputs: Union[Input, List[Input]],
            outputs: Union[Module, List[Module]],
    ): 
        pass

    @abstractmethod 
    def forward(
            self,
            x: ArrayOrTuple
    ) -> ArrayOrTuple: 
        pass

    def _map_modules(self):
        self.module_map: Dict[str, Module] = {}
        counts: Dict[str, int] = {}

        for m in self.modules:
            base = type(m).__name__
            idx = counts.get(base, 0)
            counts[base] = idx + 1
            name = f"{base}_{idx}"
            self.module_map[name] = m

    def _init_params(self):
        self.clear()
        self.eval()
        buffers = {m: [] for m in self.modules}

        for inp in self.inputs:
            sample = inp.generate_sample()
            buffers[inp].append(sample)

        for m in self.modules:
            if isinstance(m, Input):
                xi = buffers[m][0]
            else:
                xs = buffers[m]
                if len(xs) == 1:
                    xi = xs[0]
                else:
                    xi = tuple(xs)

            if len(m.prev) == 1 and isinstance(xi, tuple):
                xi = xi[0]

            m.set_input(xi)

            if m.has_params:
                self.initializer.init(m)

            yi = m.forward(xi)

            if isinstance(yi, tuple):
                for child, yj in zip(m.next, yi):
                    buffers[child].append(yj)
            else:
                for child in m.next:
                    buffers[child].append(yi)

        # Clear buffers
        for m in self.modules:
            buffers[m] = []

        self.eval()

    def get_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        state: Dict[str, Dict[str, np.ndarray]] = {}
        for name, m in self.module_map.items():
            if m.has_params:
                state[name] = {k: v.copy() for k, v in m.params.items()}
        return state

    def load_params(self, state: Dict[str, Dict[str, np.ndarray]]):
        for name, params in state.items():
            if name in self.module_map:
                m = self.module_map[name]
                if m.has_params:
                    m.load_params(params)

    def train(self):
        for m in self.modules:
            m.train()

    def eval(self):
        for m in self.modules:
            m.eval()

    def clear(self):
        for m in self.modules:
            m.clear()

    @property 
    def type(self) -> str: 
        return self._type