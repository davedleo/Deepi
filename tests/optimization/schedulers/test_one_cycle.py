import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.one_cycle import OneCycle


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

def test_one_cycle_initialization():
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

    sched = OneCycle(
        optimizer=opt,
        max_lr=1.0,
        total_steps=10,
        pct_start=0.3,
        final_div_factor=100.0,
    )

    assert sched.optimizer is opt
    assert sched._type == "scheduler.one_cycle"
    assert sched.base_lr == 0.1
    assert sched.max_lr == 1.0
    assert sched.total_steps == 10
    assert sched.pct_start == 0.3
    assert sched.final_div_factor == 100.0
    assert sched.warmup_steps == 3
    assert sched.t == 0


def test_one_cycle_first_step():
    """First step should increase LR from base_lr"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=0.1, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = OneCycle(opt, max_lr=1.0, total_steps=10)

    sched.t += 1  # t = 1
    lr = sched.update()

    expected = 0.1 + (1.0 - 0.1) * 1 / sched.warmup_steps
    assert pytest.approx(lr) == expected


def test_one_cycle_reaches_max_lr():
    """LR should reach max_lr exactly at end of warmup"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=0.2, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = OneCycle(opt, max_lr=2.0, total_steps=10, pct_start=0.4)

    for _ in range(sched.warmup_steps):
        sched.t += 1
        lr = sched.update()

    assert pytest.approx(lr) == 2.0


def test_one_cycle_decay_phase():
    """LR should decrease linearly after warmup"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=1.0, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = OneCycle(opt, max_lr=3.0, total_steps=10, pct_start=0.2)

    # Advance through warmup
    for _ in range(sched.warmup_steps):
        sched.t += 1
        sched.update()

    # First decay step
    sched.t += 1
    lr = sched.update()

    final_lr = 1.0 / sched.final_div_factor
    decay_steps = sched.total_steps - sched.warmup_steps

    expected = 3.0 - (3.0 - final_lr) * 1 / decay_steps
    assert pytest.approx(lr) == expected


def test_one_cycle_final_step():
    """Final LR should be base_lr / final_div_factor"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=0.5, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = OneCycle(opt, max_lr=5.0, total_steps=8, pct_start=0.25, final_div_factor=100)

    for _ in range(8):
        sched.t += 1
        lr = sched.update()

    assert pytest.approx(lr) == 0.5 / 100


def test_one_cycle_clamps_after_total_steps():
    """LR should not change after total_steps"""
    model = build_test_model()
    model.train()

    opt = DummyOptimizer(model, lr=0.1, regularizer=None,
                         decoupled_regularization=False,
                         maximize=False, _type="dummy")

    sched = OneCycle(opt, max_lr=1.0, total_steps=5)

    for _ in range(7):
        sched.t += 1
        lr = sched.update()

    assert pytest.approx(lr) == 0.1 / sched.final_div_factor