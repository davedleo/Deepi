import pytest

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
    """Scheduler that returns a fixed learning rate"""

    def __init__(self, optimizer, lr):
        self._next_lr = lr
        super().__init__(optimizer, _type="dummy")

    def update(self) -> float:
        return self._next_lr


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
        Scheduler(None, _type="scheduler.base")


def test_scheduler_initialization_reads_optimizer_lr():
    """Scheduler should read initial lr from optimizer"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.123,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        _type="dummy"
    )

    sched = DummyScheduler(opt, lr=0.01)

    assert sched.optimizer is opt
    assert sched.lr == 0.123


def test_scheduler_step_calls_load_lr():
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

    sched = DummyScheduler(opt, lr=0.05)
    sched.step()

    assert opt.lr == 0.05


def test_scheduler_update_return_value_is_used():
    """Returned value from update() must be passed to optimizer"""
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

    sched = DummyScheduler(opt, lr=0.9)
    sched.step()

    assert opt.get_lr() == 0.9


def test_scheduler_type_property():
    """type property should return raw scheduler type"""
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

    sched = DummyScheduler(opt, lr=0.01)

    assert sched.type == "scheduler.dummy"


def test_scheduler_str_and_repr():
    """__str__ and __repr__ should capitalize type correctly"""
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

    sched = DummyScheduler(opt, lr=0.01)

    assert str(sched) == "Scheduler.Dummy"
    assert repr(sched) == "Scheduler.Dummy"