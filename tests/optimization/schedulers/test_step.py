import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.step import Step


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

def test_step_initialization():
    """Step scheduler should initialize attributes correctly"""
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

    sched = Step(opt, factor=0.33, step_size=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.step"
    assert sched.factor == 0.33
    assert sched.step_size == 5
    assert sched.t == 0


def test_step_no_decay_before_first_multiple():
    """
    t is incremented before update():
    t = 1, 2, ..., step_size-1 -> no decay
    """
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

    sched = Step(opt, factor=0.1, step_size=3)

    sched.step()  # t = 1
    assert pytest.approx(opt.lr) == 1.0

    sched.step()  # t = 2
    assert pytest.approx(opt.lr) == 1.0


def test_step_decay_at_exact_multiple():
    """
    Decay should occur when t % step_size == 0
    """
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

    sched = Step(opt, factor=0.5, step_size=3)

    sched.step()  # t = 1
    sched.step()  # t = 2
    assert pytest.approx(opt.lr) == 1.0

    sched.step()  # t = 3 → decay
    assert pytest.approx(opt.lr) == 0.5
    assert sched.t == 3


def test_step_multiple_decays():
    """Decay should apply repeatedly at each multiple of step_size"""
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

    sched = Step(opt, factor=0.1, step_size=2)

    sched.step()  # t = 1
    assert pytest.approx(opt.lr) == 1.0

    sched.step()  # t = 2 → decay
    assert pytest.approx(opt.lr) == 0.1

    sched.step()  # t = 3
    assert pytest.approx(opt.lr) == 0.1

    sched.step()  # t = 4 → decay again
    assert pytest.approx(opt.lr) == 0.01


def test_step_factor_one():
    """If factor == 1, LR should never change"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.42,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Step(opt, factor=1.0, step_size=2)

    sched.step()
    sched.step()
    sched.step()

    assert pytest.approx(opt.lr) == 0.42