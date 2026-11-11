# test_loss.py


import jax
import jax.numpy as jnp
import numpy as np

from code.losses import MSE, CrossEntropy, BinaryCrossEntropy

# Entirely written by chatgpt, to test the loss functions and their gradients using JAX's automatic differentiation.


# --------------------------------------------------
# MSE
# --------------------------------------------------

def test_mse_value():
    loss =MSE()

    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 4.0])

    expected = 1.0 / 3.0

    np.testing.assert_allclose(
        loss.forward(y_true, y_pred),
        expected,
        rtol=1e-6
    )


def test_mse_gradient():
    loss =MSE()

    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.5, 2.5, 4.0])

    analytical = loss.derivative(y_true, y_pred)

    automatic = jax.grad(
        lambda pred: loss.forward(y_true, pred)
    )(y_pred)

    np.testing.assert_allclose(
        analytical,
        automatic,
        rtol=1e-6,
        atol=1e-7
    )


# --------------------------------------------------
# Binary Cross-Entropy
# --------------------------------------------------

def test_binary_cross_entropy_value():
    loss =BinaryCrossEntropy()

    y_true = jnp.array([1.0, 0.0])
    y_pred = jnp.array([0.8, 0.2])

    expected = -np.log(0.8)

    np.testing.assert_allclose(
        loss.forward(y_true, y_pred),
        expected,
        rtol=1e-6
    )


def test_binary_cross_entropy_gradient():
    loss =BinaryCrossEntropy()

    y_true = jnp.array([1.0, 0.0, 1.0])
    y_pred = jnp.array([0.8, 0.3, 0.6])

    analytical = loss.derivative(y_true, y_pred)

    automatic = jax.grad(
        lambda pred: loss.forward(y_true, pred)
    )(y_pred)

    np.testing.assert_allclose(
        analytical,
        automatic,
        rtol=1e-6,
        atol=1e-7
    )


# --------------------------------------------------
# Multiclass Cross-Entropy
# --------------------------------------------------

def test_cross_entropy_value():
    loss =CrossEntropy()

    y_true = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    y_pred = jnp.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1]
    ])

    expected = -(np.log(0.8) + np.log(0.7)) / 2

    np.testing.assert_allclose(
        loss.forward(y_true, y_pred),
        expected,
        rtol=1e-6
    )


def test_cross_entropy_gradient():
    loss =CrossEntropy()

    y_true = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    y_pred = jnp.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1]
    ])

    analytical = loss.derivative(y_true, y_pred)

    automatic = jax.grad(
        lambda pred: loss.forward(y_true, pred)
    )(y_pred)

    np.testing.assert_allclose(
        analytical,
        automatic,
        rtol=1e-6,
        atol=1e-7
    )
