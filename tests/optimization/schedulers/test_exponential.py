import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.exponential import Exponential


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

def test_exponential_initialization():
    """Exponential scheduler initializes factor and t correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Exponential(opt, factor=0.5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.exponential"
    assert sched.factor == 0.5
    assert sched.t == 0


def test_exponential_first_update():
    """First update corresponds to t=1"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Exponential(opt, factor=0.5)
    sched.step()  # increments t -> t=1

    expected_lr = sched.lr * (sched.factor ** (sched.t - 1))  # (t-1)=0
    lr = sched.update()
    assert pytest.approx(lr) == expected_lr
    assert sched.t == 1


def test_exponential_multiple_steps():
    """Exponential decay is applied correctly over multiple steps"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    factor = 0.5
    sched = Exponential(opt, factor=factor)

    expected_lrs = []
    num_steps = 4
    for i in range(1, num_steps + 1):
        sched.step()  # increments t
        lr = sched.update()
        expected_lr = sched.lr * (factor ** (sched.t - 1))
        expected_lrs.append(expected_lr)
        assert pytest.approx(lr) == expected_lr
        assert sched.t == i


def test_exponential_factor_one():
    """If factor=1, LR should remain unchanged"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.42, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Exponential(opt, factor=1.0)
    for i in range(5):
        sched.step()
        lr = sched.update()
        assert pytest.approx(lr) == 0.42
        assert sched.t == i + 1