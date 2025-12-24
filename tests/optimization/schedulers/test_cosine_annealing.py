import pytest
from math import cos, pi

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.cosine_annealing import CosineAnnealing


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

def test_cosine_annealing_initialization():
    """Scheduler initializes attributes correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = CosineAnnealing(opt, lr_min=0.01, T_max=5)

    assert sched.optimizer is opt
    assert sched._type == "scheduler.cosine_annealing"
    assert sched.lr_min == 0.01
    assert sched.T_max == 5
    assert sched.t == 0


def test_cosine_annealing_first_step():
    """First step should compute LR correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = CosineAnnealing(opt, lr_min=0.0, T_max=5)
    sched.step()  # increments t internally
    lr = opt.lr
    expected_lr = 1.0
    assert pytest.approx(lr) == expected_lr


def test_cosine_annealing_middle_step():
    """Check LR at middle step"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = CosineAnnealing(opt, lr_min=0.0, T_max=5)

    # Advance to t=3
    sched.step()  # t=1
    sched.step()  # t=2
    lr_prev = opt.lr + 0.0

    sched.step()  # t=3
    lr = opt.lr

    expected_lr = sched.lr_min + 0.5 * (lr_prev - sched.lr_min) * (1 + cos(pi * 2 / sched.T_max))
    assert pytest.approx(lr) == expected_lr


def test_cosine_annealing_final_step():
    """t == T_max should be near lr_min"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = CosineAnnealing(opt, lr_min=0.0, T_max=5)

    for _ in range(5):
        lr_prev = opt.lr
        sched.step()
    lr = opt.lr

    # t == T_max
    expected_lr = sched.lr_min + 0.5 * (lr_prev - sched.lr_min) * (1 + cos(pi * 4 / sched.T_max))
    assert pytest.approx(lr) == expected_lr


def test_cosine_annealing_after_T_max():
    """LR continues to follow the cosine formula even after T_max"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=1.0, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = CosineAnnealing(opt, lr_min=0.0, T_max=5)

    for _ in range(7):
        sched.step()
    lr = opt.lr

    expected_lr = sched.lr_min + 0.5 * (opt.lr - sched.lr_min) * (1 + cos(pi * sched.t / sched.T_max))
    assert pytest.approx(lr) == expected_lr