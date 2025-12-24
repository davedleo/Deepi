import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.constant import Constant


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

def test_constant_initialization():
    """Constant scheduler should initialize attributes correctly"""
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

    sched = Constant(opt, factor=0.33, milestone=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.constant"
    assert sched.factor == 0.33
    assert sched.milestone == 5
    assert sched.t == 0


def test_constant_step_before_milestone():
    """LR should be multiplied by factor before milestone"""
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

    sched = Constant(opt, factor=0.5, milestone=3)

    sched.step()  # t = 1 < milestone
    assert pytest.approx(opt.lr) == 0.5
    assert sched.t == 1

    sched.step()  # t = 2 < milestone
    assert pytest.approx(opt.lr) == 0.5
    assert sched.t == 2


def test_constant_step_at_milestone():
    """LR should stop changing once t reaches milestone"""
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

    sched = Constant(opt, factor=0.5, milestone=2)

    sched.step()  # t = 1 < milestone
    assert pytest.approx(opt.lr) == 0.5

    sched.step()  # t = 2 == milestone
    assert pytest.approx(opt.lr) == 1.0

    sched.step()  # t = 3 > milestone
    assert pytest.approx(opt.lr) == 1.0


def test_constant_milestone_one():
    """If milestone == 1, LR should never be scaled"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.3,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = Constant(opt, factor=0.1, milestone=1)

    sched.step()
    assert pytest.approx(opt.lr) == 0.3

    sched.step()
    assert pytest.approx(opt.lr) == 0.3


def test_constant_factor_one():
    """If factor == 1, LR should remain unchanged"""
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

    sched = Constant(opt, factor=1.0, milestone=10)

    sched.step()
    sched.step()
    sched.step()

    assert pytest.approx(opt.lr) == 0.42