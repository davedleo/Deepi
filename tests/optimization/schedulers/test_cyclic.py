import math
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.cyclic import Cyclic


# --------------------------------------------------------------------------
# Dummy optimizer
# --------------------------------------------------------------------------

class DummyOptimizer(Optimizer):
    def direction(self, dw, buffer):
        return self.lr * dw.copy()


# --------------------------------------------------------------------------
# Model builder
# --------------------------------------------------------------------------

def build_test_model():
    inp = Input((4,))
    dense1 = Dense(8)
    relu = ReLU()
    dense2 = Dense(3)
    inp.link(dense1)
    dense1.link(relu)
    relu.link(dense2)
    return Model(inp, dense2)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def triangular_lr(t, lr_min, lr_max, step_size):
    cycle = int((1 + t / (2 * step_size)) // 1)
    x = abs(t / step_size - 2 * cycle + 1)
    scale = max(0.0, 1.0 - x)
    return lr_min + (lr_max - lr_min) * scale


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_cyclic_initialization():
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model, lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Cyclic(opt, lr_min=0.01, lr_max=0.1, step_size=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.cyclic"
    assert sched.lr_min == 0.01
    assert sched.lr_max == 0.1
    assert sched.step_size == 5
    assert sched.scale_amplitude is False
    assert sched.scaling_factor is None
    assert sched.t == 0


def test_cyclic_triangular_wave():
    """Pure triangular schedule (no amplitude scaling)"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = Cyclic(opt, lr_min=0.0, lr_max=1.0, step_size=2)

    expected = [
        0.0,  # t=1 -> t-1=0
        0.5,
        1.0,
        0.5,
        0.0,
        0.5,
    ]

    for exp in expected:
        sched.t += 1
        lr = sched.update()
        assert pytest.approx(lr) == exp


def test_cyclic_triangular2_scaling():
    """Amplitude halves every cycle"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = Cyclic(
        opt,
        lr_min=0.0,
        lr_max=1.0,
        step_size=2,
        scale_amplitude=True,
        scaling_factor=None,
    )

    # cycles of length 4
    expected = [
        0.0, 0.5, 1.0, 0.5,      # cycle 1
        0.0, 0.25, 0.5, 0.25,   # cycle 2 (half amplitude)
    ]

    for exp in expected:
        sched.t += 1
        lr = sched.update()
        assert pytest.approx(lr) == exp


def test_cyclic_exp_range_scaling():
    """Amplitude decays exponentially every step"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    gamma = 0.9
    sched = Cyclic(
        opt,
        lr_min=0.0,
        lr_max=1.0,
        step_size=2,
        scale_amplitude=True,
        scaling_factor=gamma,
    )

    for _ in range(6):
        sched.t += 1
        t = sched.t - 1
        base = triangular_lr(t, 0.0, 1.0, 2)
        expected = base * (gamma ** t)
        lr = sched.update()
        assert pytest.approx(lr) == expected


def test_cyclic_never_negative():
    """LR must never go below lr_min"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = Cyclic(opt, lr_min=0.2, lr_max=1.0, step_size=3)

    for _ in range(20):
        sched.t += 1
        lr = sched.update()
        assert lr >= 0.2