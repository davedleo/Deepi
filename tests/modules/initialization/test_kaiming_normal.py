import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.kaiming_normal import KaimingNormal


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


def test_kaiming_normal_rule_returns_correct_shape_and_range():
    gain = 1.0
    kn = KaimingNormal(gain=gain)
    shape = (1000, 1000)
    result = kn.rule(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape

    fan_in = shape[0]
    expected_std = gain / np.sqrt(fan_in)
    assert abs(np.mean(result)) < 0.01
    assert abs(np.std(result) - expected_std) < 0.01


def test_kaiming_normal_init_replaces_shapes_with_arrays():
    kn = KaimingNormal()
    dummy = DummyParam((3, 3))
    kn.init(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape


def test_kaiming_normal_str_and_repr():
    kn = KaimingNormal()
    s = str(kn)
    r = repr(kn)
    assert "Initializer.Kaiming_normal" in s
    assert "Initializer.Kaiming_normal" in r