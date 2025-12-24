import pytest
import numpy as np

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.base import Scheduler


# --------------------------------------------------------------------------
# Dummy optimizer
# --------------------------------------------------------------------------

class DummyOptimizer(Optimizer):
    def direction(self, dw, buffer):
        return self.lr * dw.copy()


# --------------------------------------------------------------------------
# Dummy scheduler
# --------------------------------------------------------------------------

class DummyScheduler(Scheduler):
    """Simple scheduler that scales lr by a constant factor"""

    def __init__(self, optimizer, factor=0.5):
        super().__init__(optimizer, _type="dummy")
        self.factor = factor

    def update(self, lr: float):
        return lr * self.factor


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

def test_scheduler_is_abstract():
    """Scheduler cannot be instantiated without update()"""
    with pytest.raises(TypeError):
        Scheduler(None, _type="base")


def test_scheduler_initialization():
    """Scheduler should store optimizer and type correctly"""
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

    sched = DummyScheduler(opt)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.dummy"


def test_scheduler_step_updates_lr():
    """step() should update optimizer lr using update()"""
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

    sched = DummyScheduler(opt, factor=0.5)
    sched.step()

    assert pytest.approx(opt.lr) == 0.05


def test_scheduler_update_receives_current_lr():
    """update() should be called with the current lr"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.01,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    called = {}

    class TrackingScheduler(Scheduler):
        def update(self, lr):
            called["lr"] = lr
            return lr

    sched = TrackingScheduler(opt, _type="tracking")
    sched.step()

    assert called["lr"] == 0.01


def test_scheduler_multiple_steps_compound():
    """Multiple scheduler steps should compound correctly"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.2,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = DummyScheduler(opt, factor=0.5)

    sched.step()
    assert pytest.approx(opt.lr) == 0.1

    sched.step()
    assert pytest.approx(opt.lr) == 0.05