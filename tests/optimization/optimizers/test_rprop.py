import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import RProp


# --------------------------------------------------------------------------
# Model builder for tests
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
    return model, inp, dense1, relu, dense2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_rprop_basic_step():
    """RProp step should update params using sign of gradient"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1)
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in opt.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                # first step now produces ±lr updates
                expected = orig_params[name][k] - np.sign(module.grads[k]) * 0.1
                assert np.allclose(v, expected)


def test_rprop_increases_eta_on_same_sign():
    """Eta should increase when gradient keeps the same sign"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1, eta_minus=0.5, eta_plus=2.0, step_max=10.0)
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    opt.step()  # first step
    # second step, same gradient sign
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])
    opt.step()

    for module_id, module in opt.modules.items():
        for k in opt.buffer[module_id]:
            eta = opt.buffer[module_id][k]["eta"]
            assert np.all(eta > 0.1)  # eta increased above initial lr
            assert np.all(eta <= opt.step_max)


def test_rprop_decreases_eta_on_sign_flip():
    """Eta should decrease and dw set to zero when gradient flips sign"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1, eta_minus=0.5, eta_plus=2.0, step_max=10.0)
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    opt.step()  # initialize buffer

    # flip gradient
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = -np.ones_like(module.params[k])
    orig_etas = {m: {k: v["eta"].copy() for k, v in opt.buffer[m].items()} for m in opt.buffer}

    opt.step()

    for module_id, module in opt.modules.items():
        for k in opt.buffer[module_id]:
            eta = opt.buffer[module_id][k]["eta"]
            assert np.all(eta <= orig_etas[module_id][k])  # eta decreased
            dw = opt.buffer[module_id][k]["dw"]
            # dw should be 0 after sign flip
            assert np.all(dw == 0.0)


def test_rprop_maximize_flag():
    """maximize=True should invert the step direction"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1, maximize=True)
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in opt.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                # maximize flips gradient → parameters increase
                assert np.all(v >= orig_params[name][k])


def test_rprop_buffer_initialized():
    """RProp should initialize dw and eta buffers correctly"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1)
    buffer = opt.get_buffer()

    for module_id, module_buffer in buffer.items():
        for param_name, buf in module_buffer.items():
            assert "dw" in buf
            assert "eta" in buf
            assert isinstance(buf["dw"], np.ndarray)
            assert isinstance(buf["eta"], np.ndarray)
            # eta should be initialized to lr
            assert np.all(buf["eta"] == 0.1)


def test_rprop_gradients_not_modified():
    """RProp should not modify gradient arrays in-place"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RProp(model, lr=0.1)
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    grad_copies = {}
    for module in model.topology:
        if module.has_params:
            grad_copies[module] = {k: v.copy() for k, v in module.grads.items()}

    opt.step()

    for module in model.topology:
        if module.has_params:
            for k, v in module.grads.items():
                assert np.allclose(v, grad_copies[module][k])