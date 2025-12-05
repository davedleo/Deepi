import pytest
import numpy as np
from deepi.modules.core import Module

class DummyModule(Module):
    def __init__(self):
        super().__init__(_type="dummy", _has_params=True)
        self.params = {"w": np.array([1.0, 2.0])}
        self.grads = {"w": np.array([0.0, 0.0])}

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * 2

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.cache = dy * 0.5
        self.grads["w"] = dy
        return self.cache

def test_module_initialization():
    mod = DummyModule()
    assert mod.type == "module.dummy"
    assert mod.has_params is True
    assert isinstance(mod.next, list)
    assert isinstance(mod.prev, list)
    assert mod.cache is None

def test_module_str_representation():
    mod = DummyModule()
    s = str(mod)
    assert s.startswith("Dummy") or s.startswith("Module")

def test_forward_backward_behavior():
    mod = DummyModule()
    x = np.array([1.0, 2.0])
    y = mod.forward(x)
    assert np.array_equal(y, x * 2)
    
    dy = np.array([0.1, 0.2])
    dx = mod.backward(dy)
    assert np.array_equal(dx, dy * 0.5)
    assert np.array_equal(mod.grads["w"], dy)

def test_module_linking():
    mod1 = DummyModule()
    mod2 = DummyModule()
    mod1.link(mod2)
    assert mod2 in mod1.next
    assert mod1 in mod2.prev

def test_train_eval_toggle():
    mod = DummyModule()
    mod.train()
    assert mod._is_training is True
    mod.eval()
    assert mod._is_training is False

def test_clear_method():
    mod = DummyModule()
    mod.cache = np.array([1.0])
    mod.grads["w"] = np.array([5.0, 5.0])
    mod.clear()
    assert mod.cache is None
    for k, v in mod.grads.items():
        assert np.all(v == 0.0)

def test_get_params():
    mod = DummyModule()
    params = mod.get_params()
    assert isinstance(params, dict)
    assert "w" in params
    assert np.array_equal(params["w"], np.array([1.0, 2.0]))

def test_load_params():
    mod = DummyModule()
    new_params = {"w": np.array([10.0, 20.0])}
    mod.load_params(new_params)
    assert np.array_equal(mod.params["w"], new_params["w"])

def test_abstract_module_instantiation():
    from deepi.modules.core import Module
    with pytest.raises(TypeError):
        Module("base", True)