from typing import Dict, List, Optional, Union, Tuple

import numpy as np

from deepi.modules import Input, Module
from deepi.modules.initialization import Initializer, KaimingUniform

ArrayOrTuple = Union[np.ndarray, Tuple[np.ndarray, ...]]


class Model:
    
    def __init__(
        self,
        inputs: Union[Input, List[Input]],
        outputs: Union[Module, List[Module]],
        initializer: Optional[Initializer] = None
    ):
        if isinstance(inputs, Input):
            inputs = [inputs]
        if isinstance(outputs, Module):
            outputs = [outputs]

        self.inputs = inputs
        self.outputs = outputs
        self.initializer = KaimingUniform() if initializer is None else initializer

        self._build()

    # --------------------------------------------------------------
    # Topology & module mapping
    # --------------------------------------------------------------
    def _build(self):
        visited = set()
        topo: List[Module] = []

        def dfs(m: Module):
            if m in visited:
                return
            visited.add(m)
            for p in m.prev:
                dfs(p)
            topo.append(m)

        for out in self.outputs:
            dfs(out)

        # Ensure all inputs are included
        for inp in self.inputs:
            if inp not in visited:
                dfs(inp)

        self._topo = topo
        self.modules = topo
        self._init_params()
        self._map_modules()

    def _map_modules(self):
        self.module_map: Dict[str, Module] = {}
        counts: Dict[str, int] = {}

        for m in self._topo:
            base = type(m).__name__
            idx = counts.get(base, 0)
            counts[base] = idx + 1
            name = f"{base}_{idx}"
            self.module_map[name] = m

    def _init_params(self):
        self.clear()
        self.eval()
        buffers = {m: [] for m in self._topo}

        # Generate dummy inputs
        for inp in self.inputs:
            sample = inp.generate_sample()
            buffers[inp].append(sample)

        for m in self._topo:
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
        for m in self._topo:
            buffers[m] = []

        self.eval()

    # --------------------------------------------------------------
    # Parameter management
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Modes
    # --------------------------------------------------------------
    def train(self):
        for m in self.modules:
            m.train()

    def eval(self):
        for m in self.modules:
            m.eval()

    def clear(self):
        for m in self.modules:
            m.clear()

    # --------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------
    def forward(self, x: ArrayOrTuple) -> ArrayOrTuple:
        buffers: Dict[Module, List[np.ndarray]] = {m: [] for m in self.modules}

        # Feed inputs
        if len(self.inputs) == 1:
            buffers[self.inputs[0]].append(x)
        else:
            for inp, xi in zip(self.inputs, x):
                buffers[inp].append(xi)

        # Topological execution
        for m in self._topo:
            if isinstance(m, Input):
                xi = buffers[m][0]
            else:
                xs = buffers[m]
                if len(xs) == 1:
                    xi = xs[0]
                else:
                    xi = tuple(xs)

            # Normalize single-input modules
            if len(m.prev) == 1 and isinstance(xi, tuple):
                xi = xi[0]

            yi = m.forward(xi)

            # Dispatch outputs
            if isinstance(yi, tuple):
                for child, yj in zip(m.next, yi):
                    buffers[child].append(yj)
            else:
                for child in m.next:
                    buffers[child].append(yi)

        # Collect outputs
        if len(self.outputs) == 1:
            return buffers[self.outputs[0]][-1]
        else:
            return tuple(buffers[o][-1] for o in self.outputs)