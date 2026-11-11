# test_backprop.py
"""
Check backpropagation against automatic differentiation.

Compares the gradients from NeuralNetwork.backward() with jax.grad of
loss_at(), for every activation, loss and regularizer used in the
experiments. While jax differentiates, jax_numpy() swaps np for jax.numpy
inside our modules, so the code checked is the same as in the experiments.

Run from code/ with:  python test/test_backprop.py

Checks
------
1. Activation derivatives (Sigmoid, ReLU, LeakyReLU) at 50 random points.
2. Regression: Linear output + MSE, small networks, 1 and 16 Runge points.
3. Classification: Softmax output + cross-entropy, 3 classes.
4. L1 and L2 regularization on a [1, 5, 4, 1] network.
5. The real sizes from config.toml, on Runge data and MNIST images.

Each network is built with build_network() and given random biases, so no
ReLU gets z = 0 exactly, where we use derivative 0 and jax uses 0.5.

Output
------
One table per check with max |diff| and the relative error
||manual - jax|| / ||jax||, then a summary table for the report. All in
float64, so a correct gradient gives relative errors around 1e-16. The script
exits with an error if any relative error is above 1e-6.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # float64, the same as numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so code/ is found

import activations, losses, network, regularizers
from activations import Sigmoid, Linear, Softmax
from losses import MSE, CrossEntropy
from network import build_network
from config import load_config
from data import generate_data, load_mnist
from metrics import rel_error
from experiments.classification import ACTIVATIONS, REGULARIZERS, one_hot
from experiments.regression import hidden_label

OUR_MODULES = (activations, losses, network, regularizers)
TOLERANCE = 1e-6  # largest relative error we accept


# ------------------------------------------
# Gradients and comparison
# ------------------------------------------


@contextmanager  # to temporarily replace np with jax.numpy in our modules
def jax_numpy():
    """
    Use jax.numpy as np inside our own modules for the duration of a with-block.

    Our functions look up np each time they are called, so replacing the
    module attribute np is enough for jax to see our code as jax code.
    jax.numpy has no errstate, which Sigmoid.forward uses, so numpy's errstate
    is lent to it. Everything is put back when the block ends, also if an
    error is raised.
    """
    for module in OUR_MODULES:
        module.np = jnp
    jnp.errstate = np.errstate
    try:
        yield
    finally:
        for module in OUR_MODULES:
            module.np = np
        del jnp.errstate


def jax_grad(f, params):
    """
    Gradient of f at params, computed by jax.

    Parameters
    ----------
    f : callable
        Scalar function of params, e.g. lambda p: net.loss_at(p, X, y).
    params : array or list[tuple]
        Point to differentiate at: an array z, or network parameters
        [(W1, b1), (W2, b2), ...].

    Returns
    -------
    Gradient with the same structure as params, e.g. [(dW1, db1), ...].
    """
    with jax_numpy():
        return jax.grad(f)(params)


def flatten(grads):
    """All entries of a gradient, [(dW1, db1), ...] or a single array, as one vector."""
    leaves = jax.tree_util.tree_leaves(grads)
    return np.concatenate([np.ravel(leaf) for leaf in leaves])


def compare(manual, auto):
    """
    Difference between our gradient and the one from jax.

    Parameters
    ----------
    manual : array or list[tuple]
        Gradient from our code (derivative() or backward()).
    auto : array or list[tuple]
        Gradient from jax_grad, same structure as manual.

    Returns
    -------
    max_diff : float
        Largest absolute difference over all entries.
    rel : float
        Relative error ||manual - auto|| / ||auto|| over all entries.
    """
    manual, auto = flatten(manual), flatten(auto)
    return np.max(np.abs(manual - auto)), rel_error(manual, auto)


# ------------------------------------------
# Networks and data
# ------------------------------------------


def with_random_biases(net):
    """
    Replace the zero biases from build_network with random normal ones.

    With zero biases, a sample where every ReLU in a layer is off gives z = 0
    exactly in the next layer. That is the kink of ReLU, where we use 0 as the
    derivative and jax uses 0.5, so the gradients would differ even though both
    are correct. Random biases keep z away from 0, and also make the check
    closer to a trained network.

    Parameters
    ----------
    net : NeuralNetwork
        Network from build_network.

    Returns
    -------
    NeuralNetwork
        The same network, with new biases. The weights are unchanged.
    """
    rng = np.random.default_rng(1)
    net.params = [(W, rng.normal(size=b.shape)) for W, b in net.params]
    return net


def runge_data(n):
    """
    n points of the noisy Runge function from generate_data.

    Returns
    -------
    X, y : np.ndarray
        Shape (n, 1) each, the column format the network expects.
    """
    x, y = generate_data(n)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def digit_data(n, n_classes=3):
    """
    Small stand-in for MNIST: n random samples with 4 features and one-hot labels.

    Real MNIST is not needed, since the check only compares two ways of
    computing the same gradient, which must agree for any input.

    Returns
    -------
    X : np.ndarray
        Inputs, shape (n, 4).
    y : np.ndarray
        One-hot targets, shape (n, n_classes).
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 4)), one_hot(rng.integers(n_classes, size=n), n_classes)


# ------------------------------------------
# The five checks
# ------------------------------------------


def check_activations():
    """
    derivative(z) of each hidden-layer activation against jax.grad of forward(z).

    The activations act elementwise, so the gradient of sum(forward(z)) with
    respect to z is the derivative at each point.

    Returns
    -------
    list[tuple]
        (activation, max |diff|, rel. error) for each activation.
    """
    z = np.random.default_rng(0).normal(size=50)
    rows = []
    for name, act_cls in ACTIVATIONS.items():
        act = act_cls()
        auto = jax_grad(lambda z: act.forward(z).sum(), z)
        rows.append((name, *compare(act.derivative(z), auto)))
    return rows


def check_regression():
    """
    Backpropagation for regression: linear output and MSE, as in parts b and c.

    Runs for every hidden activation, 0 to 3 hidden layers, and 1 or 16 Runge
    data points (3 x 4 x 2 = 24 cases).

    Returns
    -------
    list[tuple]
        (activation, hidden layers, n, max |diff|, rel. error) for each case.
    """
    rows = []
    for name, act_cls in ACTIVATIONS.items():
        for hidden in [[], [5], [5, 4], [6, 5, 4]]:
            for n in [1, 16]:  # one sample (SGD) and a mini-batch
                X, y = runge_data(n)
                layers = [1, *hidden, 1]
                acts = [act_cls() for _ in hidden] + [Linear()]
                net = build_network(layers, acts, MSE(), optimizer=None, random_state=0)
                net = with_random_biases(net)

                auto = jax_grad(lambda p: net.loss_at(p, X, y), net.params)
                rows.append(
                    (name, hidden_label(layers), n, *compare(net.backward(X, y), auto))
                )
    return rows


def check_classification():
    """
    Backpropagation for classification: softmax output and cross-entropy, as in
    parts e and f.

    backward() uses the simplified output gradient (softmax(z) - y) / n, while
    jax differentiates softmax and cross-entropy separately, so this also checks
    that simplification. Runs for every hidden activation and 0 to 2 hidden
    layers (3 x 3 = 9 cases). No hidden layers is logistic regression.

    Returns
    -------
    list[tuple]
        (activation, hidden layers, max |diff|, rel. error) for each case.
    """
    X, y = digit_data(16)
    rows = []
    for name, act_cls in ACTIVATIONS.items():
        for hidden in [[], [5], [5, 4]]:
            layers = [4, *hidden, 3]
            acts = [act_cls() for _ in hidden] + [Softmax()]
            net = build_network(
                layers, acts, CrossEntropy(), optimizer=None, random_state=0
            )
            net = with_random_biases(net)

            auto = jax_grad(lambda p: net.loss_at(p, X, y), net.params)
            rows.append(
                (name, hidden_label(layers), *compare(net.backward(X, y), auto))
            )
    return rows


def check_regularization():
    """
    Backpropagation with L1 or L2 regularization, as in parts d and e.

    The gradient of the penalty is added to the backpropagation gradient the
    same way as in NeuralNetwork.fit, and compared with jax.grad of the total
    cost loss_at + penalty. This also checks that penalty() and grad() in
    regularizers.py agree with each other, including that the biases are not
    penalized.

    Returns
    -------
    list[tuple]
        (regularizer, lambda, max |diff|, rel. error) for L1 and L2.
    """
    X, y = runge_data(16)
    n = X.shape[0]  # same n convention as in fit, see regularizers.py
    lmbda = 0.1
    rows = []
    for name, reg_cls in REGULARIZERS.items():
        acts = [Sigmoid(), Sigmoid(), Linear()]
        net = build_network([1, 5, 4, 1], acts, MSE(), optimizer=None, random_state=0)
        net = with_random_biases(net)
        regularizer = reg_cls(lmbda)

        manual = jax.tree_util.tree_map(
            lambda g, r: g + r, net.backward(X, y), regularizer.grad(net.params, n)
        )
        auto = jax_grad(
            lambda p: net.loss_at(p, X, y) + regularizer.penalty(p, n), net.params
        )
        rows.append((name, lmbda, *compare(manual, auto)))
    return rows


def check_experiment_sizes(cfg):
    """
    Backpropagation at the sizes used in the experiments, read from config.toml.

    Regression: the [regression] architectures and the largest network from
    [activation_study], on one mini-batch of Runge data. Classification: every
    width and depth from [classification], on one mini-batch of real MNIST
    images. Runs for every hidden activation (3 x 5 + 1 + 3 x 4 = 28 cases).
    With no hidden layers, the hidden activation is not used, so [784, 10]
    (logistic regression) only runs once.

    Parameters
    ----------
    cfg : dict
        Configuration from load_config().

    Returns
    -------
    list[tuple]
        (data, activation, layers, n, max |diff|, rel. error) for each case.
    """
    study = cfg["activation_study"]
    c = cfg["classification"]
    n_in, n_out = c["input_output_layers"]

    deepest = [1] + max(study["depths"]) * [max(study["widths"])] + [1]
    runge_layers = cfg["regression"]["architectures"] + [deepest]
    mnist_layers = [
        [n_in] + d * [w] + [n_out] for d in c["depths"] for w in c["widths"]
    ]
    mnist_layers = list(dict.fromkeys(map(tuple, mnist_layers)))  # depth 0 only once

    X_runge, y_runge = runge_data(cfg["training"]["batch_size"])
    X_mnist, y_mnist, _, _ = load_mnist(c["data_path"])
    X_mnist = X_mnist[: c["batch_size"]]
    y_mnist = one_hot(y_mnist[: c["batch_size"]])

    cases = [
        ("Runge", layers, Linear, MSE, X_runge, y_runge) for layers in runge_layers
    ]
    cases += [
        ("MNIST", list(layers), Softmax, CrossEntropy, X_mnist, y_mnist)
        for layers in mnist_layers
    ]

    rows = []
    for data, layers, output_cls, loss_cls, X, y in cases:
        n_hidden = len(layers) - 2
        for name, act_cls in ACTIVATIONS.items() if n_hidden else [("-", None)]:
            acts = [act_cls() for _ in range(n_hidden)] + [output_cls()]
            net = build_network(
                layers, acts, loss_cls(), optimizer=None, random_state=0
            )
            net = with_random_biases(net)

            auto = jax_grad(lambda p: net.loss_at(p, X, y), net.params)
            rows.append(
                (data, name, str(layers), len(X), *compare(net.backward(X, y), auto))
            )
    return rows


# ------------------------------------------
# Printing
# ------------------------------------------


def print_table(title, columns, rows):
    """
    Print rows as an aligned table. The last two columns are the errors,
    printed in scientific notation.
    """
    cells = [
        [str(v) for v in row[:-2]] + [f"{v:.2e}" for v in row[-2:]] for row in rows
    ]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]

    print(f"\n{title}")
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in cells:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def main():
    """Run the five checks, print one table per check and a summary table."""
    cfg = load_config()
    errors = ["max |diff|", "rel. error"]
    checks = [
        (
            "1. Activation derivatives, 50 random points",
            "Activation derivatives",
            ["activation", *errors],
            check_activations(),
        ),
        (
            "2. Regression: linear output, MSE, Runge data (parts b, c)",
            "Regression (linear + MSE)",
            ["activation", "hidden", "n", *errors],
            check_regression(),
        ),
        (
            "3. Classification: softmax output, cross-entropy, n = 16 (parts e, f)",
            "Classification (softmax + CE)",
            ["activation", "hidden", *errors],
            check_classification(),
        ),
        (
            "4. Regularization: [1, 5, 4, 1] Sigmoid network, MSE, n = 16 (parts d, e)",
            "L1 and L2 regularization",
            ["regularizer", "lambda", *errors],
            check_regularization(),
        ),
        (
            "5. Sizes used in the experiments, one mini-batch (config.toml)",
            "Experiment sizes",
            ["data", "activation", "layers", "n", *errors],
            check_experiment_sizes(cfg),
        ),
    ]

    print("Backpropagation compared with jax.grad, float64")
    for title, _, columns, rows in checks:
        print_table(title, columns, rows)

    # Largest error in each check, for the report
    summary = [
        (name, len(rows), max(r[-2] for r in rows), max(r[-1] for r in rows))
        for _, name, _, rows in checks
    ]
    print_table(
        "Summary: largest error in each check", ["check", "cases", *errors], summary
    )

    worst = max(rel for *_, rel in summary)
    if worst > TOLERANCE:
        sys.exit(f"\nFAILED: largest relative error {worst:.2e} > {TOLERANCE:.0e}")
    print(
        f"\nAll {sum(n for _, n, *_ in summary)} cases agree (relative error < {TOLERANCE:.0e})."
    )


if __name__ == "__main__":
    main()
