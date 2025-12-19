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
        initializer: Optional[Initializer] = None,
    ):
        self._type = "model"

        # Normalize inputs and outputs to lists
        if isinstance(inputs, Input):
            inputs = [inputs]
        if isinstance(outputs, Module):
            outputs = [outputs]

        self.inputs = inputs
        self.outputs = outputs
        self.modules: List[Module] = []

        if initializer is None:
            initializer = KaimingUniform()

        self._build(initializer)

    def _build(self, initializer: Initializer):
        """
        Build the model graph by performing a DFS from outputs to inputs,
        initialize parameters, and map module names.
        """
        visited_modules = set()

        def dfs(module: Module):
            if module in visited_modules:
                return
            visited_modules.add(module)
            for prev_module in module.prev:
                dfs(prev_module)
            self.modules.append(module)

        # Traverse from outputs backwards to inputs
        for output_module in self.outputs:
            dfs(output_module)

        # Ensure all inputs are included in the module list
        for input_module in self.inputs:
            if input_module not in visited_modules:
                dfs(input_module)

        self._init_params(initializer)
        self._map_modules()

    def _init_params(self, initializer: Initializer):
        """
        Initialize parameters of all modules using the given initializer.
        This is done by simulating a forward pass with dummy inputs.
        """
        self.clear()
        self.eval()

        # Buffer to hold intermediate outputs for each module
        buffers: Dict[Module, List[np.ndarray]] = {module: [] for module in self.modules}

        # Generate dummy inputs for Input modules
        for input_module in self.inputs:
            sample_input = input_module.generate_sample()
            buffers[input_module].append(sample_input)

        # Forward pass through the modules to initialize parameters
        for module in self.modules:
            if isinstance(module, Input):
                module_input = buffers[module][0]
            else:
                module_inputs = buffers[module]
                if len(module_inputs) == 1:
                    module_input = module_inputs[0]
                else:
                    module_input = tuple(module_inputs)

            # If single previous module but input is tuple, unwrap it
            if len(module.prev) == 1 and isinstance(module_input, tuple):
                module_input = module_input[0]

            module.set_input(module_input)

            if module.has_params:
                initializer.init(module)

            module_output = module.forward(module_input)

            # Distribute outputs to next modules
            if isinstance(module_output, tuple):
                for next_module, output_part in zip(module.next, module_output):
                    buffers[next_module].append(output_part)
            else:
                for next_module in module.next:
                    buffers[next_module].append(module_output)

        # Clear buffers after initialization
        for module in self.modules:
            buffers[module] = []

    def _map_modules(self):
        """
        Create a mapping from unique module names to module instances.
        Names are generated based on module class name and occurrence count.
        """
        self.modules_map: Dict[str, Module] = {}
        module_counts: Dict[str, int] = {}

        for module in self.modules:
            base_name = type(module).__name__
            count = module_counts.get(base_name, 0)
            module_counts[base_name] = count + 1
            unique_name = f"{base_name}_{count}"
            self.modules_map[unique_name] = module

    def forward(self, x: ArrayOrTuple) -> ArrayOrTuple:
        """
        Perform a forward pass through the model given input data x.
        Supports single or multiple inputs.
        """
        buffers: Dict[Module, List[np.ndarray]] = {module: [] for module in self.modules}
        outputs_buffer: Dict[Module, np.ndarray] = {}

        # Feed inputs into the buffers
        if len(self.inputs) == 1:
            buffers[self.inputs[0]].append(x)
        else:
            assert isinstance(x, tuple), "Multiple inputs require tuple input"
            for input_module, input_data in zip(self.inputs, x):
                buffers[input_module].append(input_data)

        # Execute modules in topological order
        for module in self.modules:
            if isinstance(module, Input):
                module_input = buffers[module][0]
            else:
                module_inputs = buffers[module]
                if len(module_inputs) == 1:
                    module_input = module_inputs[0]
                else:
                    module_input = tuple(module_inputs)

            module_output = module.forward(module_input)
            outputs_buffer[module] = module_output

            # Distribute outputs to next modules
            if isinstance(module_output, tuple):
                for next_module, output_part in zip(module.next, module_output):
                    buffers[next_module].append(output_part)
            else:
                for next_module in module.next:
                    buffers[next_module].append(module_output)

        # Collect outputs from the output modules
        if len(self.outputs) == 1:
            return outputs_buffer[self.outputs[0]]

        return tuple(outputs_buffer[output_module] for output_module in self.outputs)

    def backward(self, dy: ArrayOrTuple):
        """
        Perform a backward pass through the model given output gradient dy.
        Backpropagation is triggered from the output modules and is assumed
        to propagate internally through the graph.
        """
        if len(self.outputs) == 1:
            self.outputs[0].backward(dy)

        # Multiple outputs require tuple gradient input
        elif isinstance(dy, tuple): 
            for output_module, grad in zip(self.outputs, dy):
                output_module.backward(grad)
        
        else: 
            raise ValueError()

    def get_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieve a dictionary of parameters for all modules with parameters.
        The dictionary keys are module names, and values are parameter dicts.
        """
        state: Dict[str, Dict[str, np.ndarray]] = {}
        for name, module in self.modules_map.items():
            if module.has_params:
                state[name] = {param_name: param_value.copy()
                               for param_name, param_value in module.params.items()}
        return state

    def load_params(self, state: Dict[str, Dict[str, np.ndarray]]):
        """
        Load parameters from a given state dictionary into corresponding modules.
        """
        for name, params in state.items():
            if name in self.modules_map:
                module = self.modules_map[name]
                if module.has_params:
                    module.load_params(params)

    def train(self):
        """
        Set all modules to training mode.
        """
        for module in self.modules:
            module.train()

    def eval(self):
        """
        Set all modules to evaluation mode.
        """
        for module in self.modules:
            module.eval()

    def clear(self):
        """
        Clear the internal state of all modules.
        """
        for module in self.modules:
            module.clear()

    @property
    def type(self) -> str:
        return self._type