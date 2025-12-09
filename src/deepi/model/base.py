from typing import Iterable, List, Union, Dict
import numpy as np
from deepi.modules import Module, Input


class Model:

    def __init__(
        self,
        inputs: Union[Input, Iterable[Input]],
        outputs: Union[Module, Iterable[Module]]
    ):

        # Normalize inputs into lists
        self.inputs: List[Input] = (
            [inputs] if isinstance(inputs, Input) else list(inputs)
        )

        # Normalize outputs into lists
        self.outputs: List[Module] = (
            [outputs] if isinstance(outputs, Module) else list(outputs)
        )

        # internal state
        self._is_training = False

        # validate graph integrity
        self._validate_graph()

    # ------------------------------------------------------------------------- #
    # INTERNAL VALIDATION
    # ------------------------------------------------------------------------- #

    def _validate_graph(self):
        """
        Validate that model graph is non-empty and inputs have no prev modules
        and outputs have no next modules.
        """
        assert len(self.inputs) > 0, "Model must have at least one Input node."
        assert len(self.outputs) > 0, "Model must have at least one output Module."

        for inp in self.inputs:
            assert len(inp.prev) == 0, \
                "Input modules must not have previous modules."

        for out in self.outputs:
            assert len(out.next) == 0, \
                "Output modules must not have next modules."

    def forward(self, *args) -> Union[np.ndarray, tuple]:
        """
        args must match the number of inputs.
        A single input returns directly the model output.
        Multiple inputs return a tuple of outputs.
        """
        assert len(args) == len(self.inputs), \
            f"Expected {len(self.inputs)} inputs but got {len(args)}."

        # Push inputs into the graph
        outputs = []
        for inp_module, data in zip(self.inputs, args):
            out = inp_module.forward(data)
            outputs.append(out)

        # Now expand forward until outputs are reached
        # but forward calls trigger chaining automatically
        final_outputs = []
        for out_module in self.outputs:
            final_outputs.append(out_module.y)  # y must already exist

        if len(final_outputs) == 1:
            return final_outputs[0]
        return tuple(final_outputs)

    def backward(self, dy=None):
        """
        The model itself does not compute gradients.
        Instead, backward must be applied on outputs or externally after losses.
        dy can be None → gradients start from outputs explicitly.
        """
        if dy is None:
            # must start from outputs, dy = 1 for each output
            dy_struct = [None] * len(self.outputs)
        else:
            # user-provided dy must match number of outputs
            if isinstance(dy, np.ndarray):
                dy_struct = [dy]
            else:
                dy_struct = list(dy)
            assert len(dy_struct) == len(self.outputs), \
                "dy must match number of outputs."

        # propagate
        for out_module, dy_i in zip(self.outputs, dy_struct):
            out_module.backward(dy_i)

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Collect parameters from all nodes reachable from outputs.
        Keys are globally unique by module identity.
        """
        collected = {}
        visited = set()

        def dfs(node: Module):
            if node in visited:
                return
            visited.add(node)

            if node.has_params:
                for k, v in node.get_params().items():
                    collected[id(node), k] = v

            for prev in node.prev:
                dfs(prev)

        for out in self.outputs:
            dfs(out)

        return collected

    def get_grads(self) -> Dict[str, np.ndarray]:
        """
        Same traversal strategy as get_params.
        """
        collected = {}
        visited = set()

        def dfs(node: Module):
            if node in visited:
                return
            visited.add(node)

            if node.has_params:
                for k, v in node.grads.items():
                    collected[id(node), k] = v

            for prev in node.prev:
                dfs(prev)

        for out in self.outputs:
            dfs(out)

        return collected

    def clear(self):
        """Clears all caching values and gradients in the graph."""
        visited = set()

        def dfs(node: Module):
            if node in visited:
                return
            visited.add(node)
            node.clear()
            for prev in node.prev:
                dfs(prev)

        for out in self.outputs:
            dfs(out)

    def train(self):
        self._is_training = True
        self._set_flag_training(True)

    def eval(self):
        self._is_training = False
        self._set_flag_training(False)

    def _set_flag_training(self, flag: bool):
        visited = set()

        def dfs(node: Module):
            if node in visited:
                return
            visited.add(node)

            if flag:
                node.train()
            else:
                node.eval()

            for prev in node.prev:
                dfs(prev)

        for out in self.outputs:
            dfs(out)