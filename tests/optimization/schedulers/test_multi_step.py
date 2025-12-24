import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.schedulers.multi_step import MultiStep


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

def test_multistep_initialization_single_values():
    """Single factor and milestone should be converted to lists correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=0.5, milestones=3)

    assert sched.factors == [0.5]
    assert sched.milestones == [3]
    assert sched.factors_d == {3: 0.5}


def test_multistep_initialization_multiple_values():
    """Multiple factors and milestones should be stored correctly"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=[0.5, 0.1], milestones=[3, 5])

    assert sched.factors == [0.5, 0.1]
    assert sched.milestones == [3, 5]
    assert sched.factors_d == {3: 0.5, 5: 0.1}


def test_multistep_repeat_single_factor():
    """Single factor should repeat if multiple milestones"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=0.5, milestones=[2, 4, 6])

    assert sched.factors == [0.5, 0.5, 0.5]
    assert sched.factors_d == {2: 0.5, 4: 0.5, 6: 0.5}


def test_multistep_invalid_factor_milestone_length():
    """Mismatch of multiple factors and milestones should raise ValueError"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.1, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    with pytest.raises(ValueError):
        MultiStep(opt, factors=[0.5, 0.1], milestones=[2, 4, 6])


def test_multistep_update_non_milestone_returns_lr():
    """update() should return optimizer LR if t is not a milestone"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=0.5, milestones=[5])

    sched.t = 3
    lr = sched.update()
    assert lr == opt.lr


def test_multistep_update_on_milestone_applies_factor():
    """update() should multiply LR by correct factor on milestone"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=[0.5, 0.1], milestones=[2, 4])

    sched.t = 2
    lr = sched.update()
    assert pytest.approx(lr) == 0.2 * 0.5

    sched.t = 4
    lr = sched.update()
    assert pytest.approx(lr) == 0.2 * 0.1


def test_multistep_update_with_non_integer_timestep():
    """update() should handle integer conversion gracefully (t already int in this impl)"""
    model = build_test_model()
    model.train()
    opt = DummyOptimizer(model, lr=0.2, regularizer=None, decoupled_regularization=False, maximize=False, _type="dummy")

    sched = MultiStep(opt, factors=0.5, milestones=[3])

    sched.t = 3.0  # float
    lr = sched.update()
    assert pytest.approx(lr) == 0.2 * 0.5