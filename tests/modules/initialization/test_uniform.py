import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.uniform import Uniform


class DummyParam(Module):
    _has_params = True

    def __init__(self, shape):
        super().__init__("dummy")
        self.shape = shape
        self.params = {'weight': np.empty(shape)}
        self.grads = {'weight': np.empty(shape)}

    def get_params(self):
        return self.params

    def transform(self, params):
        self.params = params

    def gradients(self):
        return self.grads


def test_uniform_rule_returns_correct_shape_and_range():
    low, high = -1.0, 1.0
    init = Uniform(low, high)
    shape = (1000, 1000)
    result = init.rule(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert np.all(result >= low)
    assert np.all(result <= high)

    # Statistiche teoriche per uniforme
    expected_mean = (low + high) / 2
    expected_std = (high - low) / np.sqrt(12)
    assert abs(np.mean(result) - expected_mean) < 0.01
    assert abs(np.std(result) - expected_std) < 0.01


def test_uniform_init_replaces_shapes_with_arrays():
    init = Uniform(-2.0, 2.0)
    dummy = DummyParam((3, 3))
    init.init(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape
        assert np.all(param >= -2.0)
        assert np.all(param <= 2.0)


def test_uniform_str_and_repr():
    init = Uniform(0.0, 1.0)
    s = str(init)
    r = repr(init)
    assert s == "Initializer.Uniform"
    assert r == "Initializer.Uniform"
    assert s == r


def test_uniform_init_no_params_does_nothing():
    class NoParam(Module):
        def __init__(self):
            super().__init__("noparam")
            self.params = {}
            self.grads = {}

        def get_params(self):
            return self.params

        def transform(self, x):
            return x

        def gradients(self, dy):
            return dy

    init = Uniform(-1.0, 1.0)
    m = NoParam()
    init.init(m)
    assert m.params == {}