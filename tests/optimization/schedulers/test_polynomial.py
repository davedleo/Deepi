import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.polynomial import Polynomial


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

def test_polynomial_initialization():
    """Polynomial scheduler should initialize attributes correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=2.0, milestone=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.polynomial"
    assert sched.power == 2.0
    assert sched.milestone == 5
    assert sched.t == 0


def test_polynomial_first_step():
    """First step (t=1) should apply polynomial decay correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=1.0, milestone=5)
    sched.t = 1  # simulate first step incremented before update

    expected_factor = (1.0 - (sched.t - 1) / sched.milestone) ** sched.power
    expected_lr = expected_factor * opt.lr

    lr = sched.update()
    assert pytest.approx(lr) == expected_lr


def test_polynomial_middle_step():
    """Check polynomial decay at middle step"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=2.0, milestone=5)
    sched.t = 3  # middle step

    factor = (1.0 - (sched.t - 1) / sched.milestone) ** sched.power
    expected_lr = factor * opt.lr

    lr = sched.update()
    assert pytest.approx(lr) == expected_lr


def test_polynomial_at_milestone():
    """t >= milestone should return optimizer lr"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.5, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=2.0, milestone=5)

    # t == milestone
    sched.t = 6
    lr = sched.update()
    assert lr == opt.lr

    # t > milestone
    sched.t = 7
    lr = sched.update()
    assert lr == opt.lr


def test_polynomial_linear_decay():
    """power=1.0 should produce linear decay"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=1.0, milestone=5)
    expected_lrs = [1.0, 0.8, 0.6, 0.4, 0.2]

    for expected in expected_lrs:
        sched.step()
        lr = opt.lr
        print(lr, expected)
        assert pytest.approx(lr) == expected


def test_polynomial_quadratic_decay():
    """power=2.0 should produce quadratic decay"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = Polynomial(opt, power=2.0, milestone=5)
    expected_lrs = [
        1.0,                  # t=1
        0.64,                 # t=2: (1 - 1/5)^2 = 0.64
        0.36,                 # t=3: (1 - 2/5)^2 = 0.36
        0.16,                 # t=4
        0.04                  # t=5: last step before milestone
    ]

    for expected in expected_lrs:
        sched.step()
        lr = opt.lr
        assert pytest.approx(lr) == expected