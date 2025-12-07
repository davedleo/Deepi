import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.xavier_uniform import XavierUniform


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


def test_xavier_uniform_rule_returns_correct_shape_and_range():
    gain = 1.0
    xu = XavierUniform(gain=gain)
    shape = (1000, 1000)
    result = xu.rule(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape

    fan_in = shape[0]
    fan_out = shape[1]
    r = gain * np.sqrt(6.0 / (fan_in + fan_out))

    # Controllo dei limiti
    assert np.all(result >= -r) and np.all(result <= r)

    # Media e std teoriche di uniforme [-r, r]
    expected_mean = 0.0
    expected_std = (2 * r) / np.sqrt(12)
    assert abs(np.mean(result) - expected_mean) < 0.01
    assert abs(np.std(result) - expected_std) < 0.01


def test_xavier_uniform_init_replaces_shapes_with_arrays():
    xu = XavierUniform()
    dummy = DummyParam((3, 3))
    xu.init(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape
        fan_in = dummy.shape[0]
        fan_out = dummy.shape[1]
        r = xu.gain * np.sqrt(6.0 / (fan_in + fan_out))
        assert np.all(param >= -r)
        assert np.all(param <= r)


def test_xavier_uniform_str_and_repr():
    xu = XavierUniform()
    s = str(xu)
    r = repr(xu)
    assert "Initializer.Xavier_uniform" in s
    assert "Initializer.Xavier_uniform" in r