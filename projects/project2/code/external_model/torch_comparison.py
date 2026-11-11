# torch_comparison.py
"""
Compare our network with PyTorch on the best combination from the Task e grid
search, read from the saved classification results (no grid search needed).
"""

from metrics import accuracy
from experiments.common import make_output_dirs, load_results
from experiments.classification import (
    TASK, RESULTS_NAME, load_data, train_classifier, predict_labels,
)
from external_model.torch_model import train_torch_classifier, predict_labels_torch
import time


def best_combination(cfg, results):
    """Best (layers, activation, gamma, regularizer, lambda) from the saved results."""
    a, d, w, g, r, l = results["best_index"]
    n_in, n_out = cfg["classification"]["input_output_layers"]
    layers = [n_in] + int(results["depths"][d]) * [int(results["widths"][w])] + [n_out]
    return (
        layers,
        str(results["activations"][a]),
        float(results["gammas"][g]),
        str(results["regularizers"][r]),
        float(results["lambdas"][l]),
    )

def chosen_combination(cfg):
    """Select the MNIST model and hyperparameters for PyTorch comparison."""
    c = cfg["mnist_cv"]
    comparison = cfg["torch_comparison"]

    layers = c["architecture"]
    activation = c["activation"]
    regularizer = c["regularizer"]

    gamma = float(comparison["gamma"])
    lmbd = float(comparison["lmbda"])

    return layers, activation, gamma, regularizer, lmbd

def run(cfg) -> None:
    """Retrain the best combination with our network and with PyTorch, and compare."""
    _, result_dir = make_output_dirs(TASK)
    results = load_results(result_dir / RESULTS_NAME)
    combo = best_combination(cfg, results)
    print("Best combination:", combo)

    data = load_data(cfg)
    X_test, y_test = data["test"]

    start = time.perf_counter()
    ours, _ = train_classifier(cfg, *combo, *data["train"])
    torch_net, _ = train_torch_classifier(cfg, *combo, *data["train"])
    elapsed = time.perf_counter() - start

    print(f"Ours:    {accuracy(y_test, predict_labels(ours, X_test)):.4f}")
    print(f"PyTorch: {accuracy(y_test, predict_labels_torch(torch_net, X_test)):.4f}")
    print(f"Time taken: {elapsed:.4f} seconds")

    # Print chosen combination
    chosen = chosen_combination(cfg)

    time_chosen = time.perf_counter()
    ours_chosen, _ = train_classifier(cfg, *chosen, *data["train"])
    torch_net_chosen, _ = train_torch_classifier(cfg, *chosen, *data["train"])
    elapsed_chosen = time.perf_counter() - time_chosen

    print("\nChosen combination:", chosen)
    print(f"Ours:    {accuracy(y_test, predict_labels(ours_chosen, X_test)):.4f}")
    print(f"PyTorch: {accuracy(y_test, predict_labels_torch(torch_net_chosen, X_test)):.4f}")
    print(f"Time taken: {elapsed_chosen:.4f} seconds")



def plot(cfg) -> None:
    """Nothing to plot yet."""
