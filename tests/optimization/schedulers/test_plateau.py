import pytest
import numpy as np

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.plateau import Plateau


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

def test_plateau_initialization():
    """Plateau scheduler initializes attributes correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Plateau(opt, factor=0.5, tol=1e-3, patience=3)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.plateau"
    assert sched.factor == 0.5
    assert sched.tol == 1e-3
    assert sched.patience == 3
    assert sched.counter == 0
    assert sched.val_prev == 1e5
    assert sched.t == 0


def test_plateau_no_decay_before_patience():
    """LR should remain unchanged if val changes exceed tol before patience"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Plateau(opt, factor=0.5, tol=1e-3, patience=3)

    # values differ by more than tol -> counter reset
    for val in [1.0, 0.9, 0.85]:
        sched.step(val)
        assert opt.lr == 0.2
        assert sched.counter == 0


def test_plateau_decay_after_patience():
    """LR should decay after patience consecutive small changes"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.5, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Plateau(opt, factor=0.1, tol=0.01, patience=3)

    # First patience-1 steps: counter below patience -> no decay
    for _ in range(3):
        sched.step(1.0)
        assert opt.lr == 0.5

    # Third step: counter == patience -> decay
    sched.step(1.0)
    expected_lr = 0.5 * 0.1
    assert pytest.approx(opt.lr) == expected_lr


def test_plateau_counter_resets_on_large_change():
    """Counter resets if value change exceeds tol"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.3, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Plateau(opt, factor=0.5, tol=0.01, patience=2)

    # two steps below tol -> counter = 2
    sched.step(1.0)
    sched.step(1.0)
    assert sched.counter == 1

    # large change -> counter resets
    sched.step(1.1)
    assert sched.counter == 0
    # LR should remain unchanged because patience not reached again
    assert opt.lr == 0.3


def test_plateau_multiple_decays():
    """Scheduler can decay LR multiple times if plateau occurs repeatedly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Plateau(opt, factor=0.5, tol=1e-3, patience=2)

    # First decay
    sched.step(1.0)
    sched.step(1.0)  # counter=1
    sched.step(1.0)  # counter=2 -> decay
    assert pytest.approx(opt.lr) == 0.5

    # Repeat same small values -> second decay
    sched.step(1.0)
    sched.step(1.0)  # counter=1
    sched.step(1.0)  # counter=2 -> decay
    assert pytest.approx(opt.lr) == 0.25