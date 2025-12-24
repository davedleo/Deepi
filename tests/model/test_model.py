import numpy as np
import pytest
from deepi.model import Model
from deepi.modules import Input, Dense, Add

# Helper: Small, deterministic input for clarity
def small_input(shape):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

# -------------------------------
# Forward Pass
# -------------------------------
def test_forward_pass_dense_chain():
    """Test that Model.forward computes correct output shape and values for a simple chain."""
    x = small_input((2, 3))  # batch size 2, 3 features
    inp = Input((3,))
    dense = Dense(2)
    inp.link(dense)
    model = Model(inp, dense)
    out = model(x)
    # Output should have shape (2, 2)
    assert out.shape == (2, 2)
    # Output should be computed as x @ dense.params["w"] + dense.params["b"]
    expected = x @ dense.params["w"] + dense.params["b"]
    np.testing.assert_allclose(out, expected)

# -------------------------------
# Backward Propagation
# -------------------------------
def test_backward_pass_dense_chain():
    """Test that Model.backward propagates gradients through all modules."""
    x = small_input((2, 4))
    inp = Input((4,))
    dense1 = Dense(3)
    dense2 = Dense(2)
    inp.link(dense1)
    dense1.link(dense2)
    model = Model(inp, dense2)
    model.train()
    out = model(x)
    dout = np.ones_like(out)
    dense2.backward(dout)
    # Dense layers should have non-None gradients
    for m in [dense1, dense2]:
        assert hasattr(m, "grads")
        assert m.grads["w"] is not None
        # Gradient shapes: for Dense, grad matches params["w"]
        assert m.grads["w"].shape == m.params["w"].shape

# -------------------------------
# Training/Eval Mode and Clearing
# -------------------------------
def test_train_eval_clear_modes():
    """Test that Model.train(), Model.eval(), and Model.clear() propagate to all modules."""
    inp = Input((2,))
    dense = Dense(2)
    inp.link(dense)
    model = Model(inp, dense)
    # Train mode
    model.train()
    for m in model.topology:
        assert getattr(m, "_is_training", None) is True
    # Eval mode
    model.eval()
    for m in model.topology:
        assert getattr(m, "_is_training", None) is False
    # Clear gradients
    model.train()
    model(small_input((1, 2)))
    model.backward(np.ones((1, 2)))
    model.clear()
    for m in model.topology:
        if hasattr(m, "grads"):
            assert m.grads["w"] is None or np.allclose(m.grads["w"], 0)

def test_get_and_load_params():
    """Test Model.get_params and Model.load_params for correctness."""
    inp = Input((2,))
    dense = Dense(2)
    inp.link(dense)
    model = Model(inp, dense)

    # Get current parameters and modify them
    params = model.get_params()
    params["Dense_0"]["w"] = np.ones_like(params["Dense_0"]["w"])
    params["Dense_0"]["b"] = np.ones_like(params["Dense_0"]["b"])

    # Load modified params into a new model with the same architecture
    inp2 = Input((2,))
    dense2 = Dense(2)
    inp2.link(dense2)
    model2 = Model(inp2, dense2)
    model2.load_params(params)

    loaded = model2.get_params()

    # Compare loaded parameters
    np.testing.assert_allclose(loaded["Dense_0"]["w"], params["Dense_0"]["w"])
    np.testing.assert_allclose(loaded["Dense_0"]["b"], params["Dense_0"]["b"])

# -------------------------------
# Topological Order
# -------------------------------
def test_model_topological_order():
    """Test that Model.topology returns modules in correct topological order."""
    inp = Input((3,))
    dense1 = Dense(4)
    dense2 = Dense(2)
    inp.link(dense1)
    dense1.link(dense2)
    model = Model(inp, dense2)
    mods = model.topology
    assert mods == [inp, dense1, dense2]

def test_model_topological_order_residual():
    """Test that Model.topology returns correct topological order for a residual connection."""
    inp = Input((3,))
    dense1 = Dense(3)
    dense2 = Dense(2)
    add = Add()

    inp.link(dense1)
    inp.link(add)
    dense1.link(add)
    add.link(dense2)

    model = Model(inp, dense2)
    mods = model.topology

    # Valid topological order: inp must come first,
    # dense1 before add, add before dense2
    assert mods[0] is inp
    assert dense1 in mods
    assert add in mods
    assert dense2 in mods

    assert mods.index(inp) < mods.index(dense1)
    assert mods.index(inp) < mods.index(add)
    assert mods.index(dense1) < mods.index(add)
    assert mods.index(add) < mods.index(dense2)
