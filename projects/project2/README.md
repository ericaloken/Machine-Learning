# Classification and Regression, from Linear and Logistic Regression to Neural Networks

Project 2 in FYS-STK3155/FYS-STK4155, University of Oslo, autumn 2026.

## Description

We write our own feed-forward neural network (FFNN) with backpropagation and
apply it to the noisy Runge function from Project 1, and then to image
classification. The parts are:

- **Regression (b):** FFNN on the Runge function, with plain gradient descent,
  RMSProp and Adam, compared with the closed-form OLS benchmark from Project 1.
- **Architecture and activations (c):** sigmoid, ReLU and leaky ReLU over a
  grid of depths and widths.
- **Regularization (d):** L1 and L2 penalties over a grid of $\lambda$ and
  learning rate, compared with Ridge and Lasso from Project 1.

All parameters are set in `code/config.toml` and the random seed is 67
everywhere. Every figure and printed table can be reproduced with
`python3 code/main.py machine.learn`.

**Remark**: This file has been written with great help of Artificial
Intelligence based on input from us, the project description and the files
inside the project. It has however been verified and edited for better wording
and better fit with our solution.

---

## Methods

**Network** (`network.py`). `NeuralNetwork` takes a layer list
(e.g. `[1, 50, 50, 1]`), one activation object per layer after the input, a
loss, an optimizer and an optional regularizer. Weights use He initialization
for ReLU/leaky ReLU and Xavier otherwise, and the biases alwatys start at zero. Parameters are
stored as a list of `(W, b)` pairs and updated as a JAX pytree, so the same
optimizer code works for any depth and also for linear regression with parameter `theta`. `fit` reshuffles the training data every
epoch and supports full-batch (`batch_size=None`) and mini-batch training.
Backpropagation is implemented by hand, with the softmax + cross-entropy output
delta simplified to $a - y$.

**Activations and losses** (`activations.py`, `losses.py`). Sigmoid, ReLU, leaky
ReLU, linear and softmax; MSE, cross-entropy and binary cross-entropy. Each is
a class with `forward` and `derivative`, behind an abstract base class.

**Optimizers** (`optimizers.py`). One `Optimizer` base class with subclasses for
plain GD, momentum, AdaGrad, RMSProp and Adam. All support an optional inverse
time decay schedule, and are selected by name through `make_optimizer`.

**Regularization** (`regularizers.py`). L1 and L2 penalties act on the weights
only, never on the biases. The penalty is scaled by the full training-set size,
so mini-batch gradients stay unbiased:

$$
C(\theta) = L(\theta) + \frac{\lambda}{n}R(W), \qquad
R = \lVert W\rVert_1 \ (\text{L1}), \quad \lVert W\rVert_2^2 \ (\text{L2}).
$$

**Linear baselines** (`linear_models.py`). OLS and Ridge in closed form from the
SVD, plus Lasso from Scikit-Learn, reused from Project 1 with the same
$\lambda$ convention.

**Data** (`data.py`). Noisy Runge data, polynomial design matrix, train/test
split and a `Scaler` that standardizes $x$ and $y$ on the training set. Every
experiment uses a train/validation/test split: train fits the weights,
validation selects hyperparameters, test is used once per selected model.

---

## Project structure

```text
project2/
├── README.md
├── PROJECT_PLAN.md
├── supplementery_figs.ipynb
├── code/
│   ├── main.py              # CLI entry point (argparse + argcomplete)
│   ├── config.py            # reads config.toml
│   ├── config.toml          # all parameters
│   ├── network.py           # FFNN and backpropagation
│   ├── activations.py       # sigmoid, ReLU, leaky ReLU, linear, softmax
│   ├── losses.py            # MSE, cross-entropy, binary cross-entropy
│   ├── optimizers.py        # GD, momentum, AdaGrad, RMSProp, Adam
│   ├── regularizers.py      # L1, L2 (weights only)
│   ├── linear_models.py     # OLS, Ridge, Lasso baselines from project 1
│   ├── data.py              # Runge data, splitting, scaling
│   ├── metrics.py           # MSE, R2, relative error
│   ├── plotting_utils.py    # shared plotting helpers
│   ├── experiments/         # one module per task, each with run() and plot()
│   └── test/                # pytest unit tests
└── output/                  # one subfolder per task
    └── <task>/
        ├── figures/         # pdf, with pngs/ for the README
        └── results/         # <task>.npz
```

---

## Installation

The code requires **Python 3.11 or newer** (it uses `tomllib`) and

```text
numpy  scipy  matplotlib  seaborn  scikit-learn  jax  jaxlib  joblib  argcomplete
```

All dependencies are listed in `requirements.txt` at the root of the repository:

```bash
# from the repository root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The plots use the shared style file `style.mplstyle` in the repository root, so
the project must be run from inside the cloned repository.

---

## Usage

Run one or more tasks from `code/`:

```bash
python3 main.py regression                 # train, save results and plot
python3 main.py regression plot            # only plot from saved results
python3 main.py activation_study classification
python3 main.py                            # run all tasks
python3 main.py machine.learn              # also runs all tasks
python3 main.py 'machine.learn()'          # same, quotes needed because of the parentheses
```

Tab completion for task names works via `argcomplete` (`pip install argcomplete`,
then `activate-global-python-argcomplete --user`).

| Task | Part | Output |
|------|------|--------|
| `regression` | b| FFNN vs OLS on the Runge function; learning-rate heatmaps per optimizer, loss curves, test MSE, fit |
| `activation_study` | c | Train/validation MSE vs depth and width for sigmoid, ReLU and leaky ReLU.|
| `regularization_study` | d | MSE heatmaps over $\lambda$ and learning rate for L1 and L2. Comparison with Ridge and Lasso from Project 1 |
| `classification` | e | MNIST accuracy vs $\lambda$ and learning rate.|

Results are written to `output/<task>/results/<task>.npz` and figures to
`output/<task>/figures/`. The hyperparameter grids are trained in parallel with
`joblib`; set the number of workers with `n_jobs` in `config.toml`.

To save the printed output of a full run:

```bash
python3 main.py 2>&1 | tee output/output_all_tasks.txt
```

---

## Declaration of use of generative AI

We used large language models (ChatGPT and Claude) as an aid during the project.
They helped with debugging, with suggestions for code structure, with the
plotting and table-printing code, and with writing this README. All suggestions
were reviewed, tested and adapted by us before being included. We are
responsible for all code, results and conclusions. The full declaration,
following the course guidelines, is given in the report.

---

## Authors

Erica Løken, Caspper Johnsen, Adrian Gundersen, Victor Berge Johansen
