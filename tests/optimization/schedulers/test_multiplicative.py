import pytest
import numpy as np

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.multiplicative import Multiplicative


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

def test_multiplicative_initialization():
    """Scheduler should initialize lambda and timestep correctly"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Multiplicative(opt, lmbd=lambda t: 0.9)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.multiplicative"
    assert sched.t == 0


def test_multiplicative_single_step():
    """Learning rate should be multiplied by lambda(1)"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Multiplicative(opt, lmbd=lambda t: 0.5)
    sched.step()

    assert pytest.approx(opt.lr) == 0.05
    assert sched.t == 1


def test_multiplicative_multiple_steps():
    """Learning rate should compound multiplicatively over steps"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=1.0,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Multiplicative(opt, lmbd=lambda t: 0.5)

    sched.step()
    assert pytest.approx(opt.lr) == 0.5
    assert sched.t == 1

    sched.step()
    assert pytest.approx(opt.lr) == 0.25
    assert sched.t == 2

    sched.step()
    assert pytest.approx(opt.lr) == 0.125
    assert sched.t == 3


def test_multiplicative_lambda_receives_correct_timestep():
    """Lambda function should receive correct timestep t"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    calls = []

    def lmbd(t):
        calls.append(t)
        return 1.0

    sched = Multiplicative(opt, lmbd=lmbd)

    sched.step()
    sched.step()
    sched.step()

    assert calls == [1, 2, 3]