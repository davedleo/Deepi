import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.xavier_normal import XavierNormal


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


def test_xavier_normal_rule_returns_correct_shape_and_std():
    gain = 1.0
    xn = XavierNormal(gain=gain)
    shape = (1000, 1000)
    result = xn.rule(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape

    fan_in = shape[0]
    fan_out = shape[1]
    expected_std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    assert abs(np.mean(result)) < 0.01
    assert abs(np.std(result) - expected_std) < 0.01


def test_xavier_normal_init_replaces_shapes_with_arrays():
    xn = XavierNormal()
    dummy = DummyParam((3, 3))
    xn.init(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape


def test_xavier_normal_str_and_repr():
    xn = XavierNormal()
    s = str(xn)
    r = repr(xn)
    assert "Initializer.Xavier_normal" in s
    assert "Initializer.Xavier_normal" in r