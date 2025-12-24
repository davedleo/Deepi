import pytest
from math import cos, pi

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestarts,
)


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
# Tests
# --------------------------------------------------------------------------

def test_cosine_annealing_warm_restarts_initialization():
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy",
    )

    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, lr_min=0.01)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.cosine_annealing_warm_restarts"
    assert sched.T_0 == 5
    assert sched.T_mult == 2
    assert sched.lr_min == 0.01
    assert sched.T_i == 5
    assert sched.T_cur == 0
    assert sched.t == 0


def test_first_step_returns_max_lr():
    """First update should return the current (max) LR"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = CosineAnnealingWarmRestarts(opt, T_0=4, lr_min=0.0)

    sched.t += 1
    lr = sched.update()

    assert pytest.approx(lr) == 1.0


def test_cosine_decay_within_first_cycle():
    """LR should follow cosine decay inside a cycle"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = CosineAnnealingWarmRestarts(opt, T_0=4, lr_min=0.0)

    expected_lrs = []
    for step in range(4):
        expected = 0.0 + 0.5 * (1.0 - 0.0) * (
            1.0 + cos(pi * step / 4)
        )
        expected_lrs.append(expected)

    for expected in expected_lrs:
        sched.t += 1
        lr = sched.update()
        assert pytest.approx(lr) == expected


def test_restart_resets_cycle_position():
    """After T_0 steps, the scheduler should restart"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=2.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = CosineAnnealingWarmRestarts(opt, T_0=3, lr_min=0.0)

    # Complete first cycle
    for _ in range(3):
        sched.t += 1
        sched.update()

    # First step after restart
    sched.t += 1
    lr = sched.update()

    assert sched.T_cur == 1
    assert sched.T_i == 3
    assert pytest.approx(lr) == 2.0


def test_cycle_length_increases_with_T_mult():
    """Cycle length should grow when T_mult > 1"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2, lr_min=0.0)

    # First cycle (length 2)
    for _ in range(2):
        sched.t += 1
        sched.update()

    assert sched.T_i == 2

    # Trigger restart
    sched.t += 1
    sched.update()

    assert sched.T_i == 4  # doubled


def test_lr_never_below_lr_min():
    """LR should never drop below lr_min"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = CosineAnnealingWarmRestarts(opt, T_0=3, lr_min=0.3)

    for _ in range(20):
        sched.t += 1
        lr = sched.update()
        assert lr >= 0.3