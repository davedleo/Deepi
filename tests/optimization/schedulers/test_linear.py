import pytest
import numpy as np

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.linear import Linear


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

    model = Model(inp, dense2)
    return model


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_linear_initialization():
    """Linear scheduler should create factors and store milestone correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Linear(opt, start_factor=0.5, end_factor=1.0, milestone=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.linear"
    assert sched.milestone == 5
    assert np.allclose(sched.factors, np.linspace(0.5, 1.0, 5))


def test_linear_update_before_milestone():
    """update() should scale LR according to factors before milestone"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Linear(opt, start_factor=0.5, end_factor=1.0, milestone=4)

    # t = 1
    sched.t = 1
    lr1 = sched.update()
    assert pytest.approx(lr1) == 0.2 * sched.factors[0]

    # t = 2
    sched.t = 2
    lr2 = sched.update()
    assert pytest.approx(lr2) == 0.2 * sched.factors[1]

    # t = 4
    sched.t = 4
    lr4 = sched.update()
    assert pytest.approx(lr4) == 0.2 * sched.factors[3]


def test_linear_update_after_milestone():
    """update() should return optimizer LR unchanged after milestone"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.3, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Linear(opt, start_factor=0.5, end_factor=1.0, milestone=3)

    sched.t = 4  # t > milestone
    lr = sched.update()
    assert lr == opt.lr


def test_linear_full_progression():
    """Simulate multiple steps and check LR progression"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Linear(opt, start_factor=0.5, end_factor=1.0, milestone=5)

    expected_lrs = [1.0 * f for f in sched.factors]

    for step, expected_lr in enumerate(expected_lrs, start=1):
        sched.t = step
        lr = sched.update()
        assert pytest.approx(lr) == expected_lr

    # After milestone, LR stays constant
    sched.t = 6
    lr_after = sched.update()
    assert lr_after == opt.lr