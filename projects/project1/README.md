# Regression Analysis, Resampling Methods and Gradient Descent

Project 1 in FYS-STK3155/FYS-STK4155, University of Oslo, autumn 2026.

## Description

We fit polynomials to Runge's function and study:

- **Closed-form regression:** Ordinary Least Squares (OLS) and Ridge regression, solved with the SVD (parts a–b).
- **Resampling:** the bias–variance tradeoff with the bootstrap (part c), and $k$-fold cross-validation (part d).
- **Gradient descent:** our own gradient descent with analytical and automatically differentiated (JAX) gradients (part e).
- **Adaptive optimizers:** momentum, AdaGrad, RMSProp and Adam (part f).
- **Lasso regression:** solved with our own (sub)gradient descent and compared with Scikit-Learn (part g).
- **Stochastic gradient descent:** mini-batches, epochs and learning-rate schedules (part h).
- **Final model selection:** OLS, Ridge and Lasso compared with cross-validation (part i).

All parameters are set in `config.toml`. Every figure and every printed table in the report can be reproduced with `python3 main.py`.

**Remark**: This file has been written with great help of Artificial Intelligence based on input from us, the project description and the files inside the project. It has however been verified and edited for better wording and better fit with our solution.

---

## Problem setup

The data are generated from the one-dimensional Runge function

$$
f(x)=\frac{1}{1+25x^2}, \qquad x\in[-1,1],
$$

with $x$ drawn uniformly from $[-1, 1]$ and normally distributed noise,

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

with $\sigma = 0.1$ by default. The design matrix is the polynomial (Vandermonde) matrix with columns $1, x, x^2, \dots, x^d$, where $d$ is the polynomial degree.

---

## Methods

### Preprocessing

The data are split into a training set (80 %) and a test set (20 %). The non-intercept columns of $X$ and the targets $y$ are standardized with the mean and standard deviation of the **training data only**. The same transformation is then applied to the test data. Inside cross-validation the scaling is refitted on the training folds of each split, because it happens in `Model.fit` (`models.py`).

Since the features and targets are centered, the coefficient of the intercept column is zero for every method. Predictions are transformed back to the original scale of $y$.

### Cost functions and the $\lambda$ convention

All methods minimize

$$
C(\boldsymbol\theta) = \frac{1}{n}\lVert \boldsymbol y - X\boldsymbol\theta\rVert_2^2 + \frac{\lambda}{n} P(\boldsymbol\theta),
\qquad
P = \begin{cases} 0 & \text{OLS} \\ \lVert\boldsymbol\theta\rVert_2^2 & \text{Ridge} \\ \lVert\boldsymbol\theta\rVert_1 & \text{Lasso,} \end{cases}
$$

on the standardized data. With this convention the Ridge minimizer is the familiar $(X^TX+\lambda I)^{-1}X^T\boldsymbol y$. The gradients are

$$
\nabla C_{\text{Ridge}} = \tfrac{2}{n}X^T(X\boldsymbol\theta-\boldsymbol y) + \tfrac{2\lambda}{n}\boldsymbol\theta,
\qquad
\nabla C_{\text{Lasso}} = \tfrac{2}{n}X^T(X\boldsymbol\theta-\boldsymbol y) + \tfrac{\lambda}{n}\,\mathrm{sign}(\boldsymbol\theta),
$$

and the Hessian is $\tfrac{2}{n}X^TX$, plus $\tfrac{2\lambda}{n}I$ for Ridge.

Scikit-Learn's `Lasso` minimizes $\frac{1}{2n}\lVert\boldsymbol y - X\boldsymbol\theta\rVert^2 + \alpha\lVert\boldsymbol\theta\rVert_1$, so $\alpha = \lambda/(2n_\text{train})$.

### Closed-form OLS and Ridge

Both are computed from the SVD $X = U\Sigma V^T$, without forming or inverting $X^TX$:

$$
\hat{\boldsymbol\theta}_\text{OLS} = V\Sigma^{+}U^T\boldsymbol y,
\qquad
\hat{\boldsymbol\theta}_\text{Ridge} = V\,\mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)U^T\boldsymbol y .
$$

Singular values below the same relative tolerance as `np.linalg.pinv` are treated as zero.

### Gradient descent and optimizers

`optimizers.general_optimizer` implements one update rule that reduces to plain GD, momentum, AdaGrad, RMSProp and Adam through its parameters. Every method can be run full-batch or with mini-batches (SGD). In SGD the data are reshuffled every epoch, and an optional learning-rate schedule (inverse time decay) can be used.

Gradients are computed either analytically (`gradients.py`) or by automatic differentiation with `jax.grad`. The penalty term is scaled with the full training-set size, so that mini-batch gradients are unbiased.

Lasso uses the subgradient $\mathrm{sign}(\theta_j)$, which we set to $0$ at $\theta_j = 0$. JAX returns $+1$ there, which is also a valid subgradient.

### Resampling

- **Bootstrap** (`resampling.bias_variance_bootstrap`): models are fitted to resampled training sets and evaluated on a fixed test set. This gives the test MSE, the variance and the squared bias. The bias is measured either against the noisy $y$ (so that MSE $=$ bias$^2$ + variance exactly) or against the true $f(x)$ (so that MSE $\approx$ bias$^2$ + variance + $\sigma^2$).
- **Cross-validation:** `KFold(shuffle=True)` with `cross_val_score`, using a Scikit-Learn-compatible wrapper around our own models (`models.RegressionEstimator`). Our own $k$-fold loop (`resampling.cross_validation_manual`) reproduces Scikit-Learn exactly when given the same folds (`python3 main.py verify_cv`).

---

## Project structure

```text
project1/
├── main.py                     # Entry point: runs all or selected tasks
├── config.toml                 # All parameters (data, degrees, lambdas, optimizers, ...)
├── config.py                   # Loads config.toml
├── data.py                     # Runge function, data generation, design matrix, train/test split
├── metrics.py                  # MSE, R², relative error
├── models.py                   # Model class: scaling, closed-form OLS/Ridge (SVD), iterative fits,
│                               #   Scikit-Learn wrapper (RegressionEstimator)
├── gradients.py                # Cost functions, analytical gradients and JAX (AD) gradients
├── optimizers.py               # GD, SGD, momentum, AdaGrad, RMSProp, Adam; Hessian learning-rate bound
├── resampling.py               # Bootstrap bias-variance and k-fold cross-validation (own and Scikit-Learn)
├── analysis.py                 # RungeRegression class: data + fitting/evaluation used by parts a-c
├── plotting.py                 # All figures (saved as PDF in figures/ and PNG in figures/pngs/)
├── experiments/
│   ├── common.py               # Shared helpers (model setup, optimizer settings, Scikit-Learn Lasso)
│   ├── ols.py                  # Part a: OLS
│   ├── ridge.py                # Part b: Ridge
│   ├── bootstrap.py            # Part c: bias-variance with the bootstrap
│   ├── cross_validation.py     # Part d: cross-validation, comparison with the bootstrap
│   ├── gradient_descent.py     # Part e: GD with analytical and AD gradients
│   ├── adaptive_gd.py          # Part f: momentum, AdaGrad, RMSProp, Adam
│   ├── lasso.py                # Part g: Lasso
│   ├── stochastic_gd.py        # Part h: stochastic gradient descent
│   └── model_selection.py      # Part i: final model selection
├── tests/
│   ├── test_cross_validation.py  # Own k-fold CV vs Scikit-Learn (part d)
│   └── test_gradient.py          # Analytical vs AD gradients (part e)
├── results/
│   └── output_all_tasks.txt    # Printed output from a full run
├── figures/                    # Generated figures (PDF, not tracked by git)
│   └── pngs/                   # Same figures as PNG
└── supplementary_figures.ipynb # Supplementary figures referenced from the report
```

---

## Installation

The code requires **Python 3.11 or newer** (it uses `tomllib`) and the packages

```text
numpy  scipy  matplotlib  scikit-learn  jax  jaxlib
```

All dependencies are listed in `requirements.txt` at the root of the repository:

```bash
# from the repository root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The plots use the shared style file `style.mplstyle` in the repository root, so the project must be run from inside the cloned repository.

---

## Usage

Run all tasks from `projects/project1/`:

```bash
python3 main.py
```

A full run takes about 3 minutes on a mediocre laptop for our particular config. Figures are written to `figures/` (PDF) and `figures/pngs/` (PNG), and tables are printed to the terminal.

Run one or more tasks by name:

```bash
python3 main.py ols ridge
```

| Task               | Part | Output |
|--------------------|------|--------|
| `ols`              | a    | MSE and $R^2$ vs degree for several $n$ and $\sigma$; coefficients vs degree |
| `ridge`            | b    | MSE and $R^2$ vs $\lambda$ and vs degree; coefficients; shrinkage factors of the singular values |
| `bootstrap`        | c    | Train/test MSE vs degree (cf. Hastie et al., Fig. 2.11); bias$^2$, variance and MSE vs degree and $n$ |
| `cv`               | d    | CV MSE for OLS and Ridge ($k=5, 10$) vs degree and $\lambda$; CV vs bootstrap |
| `verify_cv`        | d    | Table: own $k$-fold CV vs Scikit-Learn with identical folds |
| `gd`               | e    | Table: GD vs closed form, analytical vs AD gradient, learning rate vs Hessian bound |
| `verify_gradients` | e    | Table: max difference between analytical and AD gradients |
| `adaptive_gd`      | f    | Iterations to target accuracy and final error vs learning rate; convergence curves |
| `lasso`            | g    | Subgradient at $\theta=0$; Lasso vs $\lambda$ (own GD and Scikit-Learn); optimizers; OLS/Ridge/Lasso coefficients |
| `sgd`              | h    | Error vs mini-batch size, SGD vs full batch, constant vs decaying learning rate, wall time |
| `model_selection`  | i    | Best model per method from 5-fold CV; CV MSE vs degree; Ridge and Lasso heatmaps; bias-variance table |

To save the printed output of a full run:

```bash
python3 main.py 2>&1 | tee results/output_all_tasks.txt
```

---

## Parameters used in the report

All values are read from `config.toml`, and the random seed is 67 everywhere. The most important settings are:

| Part | Setting |
|------|---------|
| all  | $n = 100$, $\sigma = 0.1$, uniform $x$, test size 0.2 |
| a    | Degrees 1–15; $n \in \{50, 100, 500\}$; $\sigma \in \{0, 0.1, 0.2\}$ |
| b    | Degree 15 with $\lambda = 10^{-8}, \dots, 10^{4}$ (13 values); degrees 1–15 with $\lambda \in \{10^{-4}, 10^{-2}, 1, 100\}$ |
| c    | $n = 500$, degrees 1–30, 100 bootstrap samples, $n \in \{50, 100, 500\}$ |
| d    | $n = 500$, degrees 1–30, $k \in \{5, 10\}$; Ridge at degree 15 ($\lambda$ sweep) and at $\lambda = 0.01$ (degree sweep) |
| e    | Degree 5, $\lambda = 1$, $\gamma = 0.8\,\gamma_\text{max}$ with $\gamma_\text{max} = 2/\lambda_\text{max}(H)$, at most $10^4$ iterations |
| f    | Learning rates $10^{-3}$–$10$; target relative error $10^{-3}$; momentum $\beta = 0.9$, RMSProp $\rho = 0.9$, Adam $(\beta_1,\beta_2) = (0.9, 0.999)$ |
| g    | Degree 5, $\lambda = 1$; $\lambda = 10^{-3}, \dots, 10^{2}$ (11 values) |
| h    | 1000 epochs, $M \in \{1, 5, 10, 20, 50\}$ and full batch; $M = 10$ for the epoch and schedule studies; learning rate halved after 100 epochs |
| i    | 5-fold CV on the part-d data; OLS/Ridge degrees 1–30, Lasso degrees 1–10; Ridge $\lambda = 10^{-10}, \dots, 10^{2}$, Lasso $\lambda = 10^{-4}, \dots, 10^{2}$ |

Notes:

- `test_size` in `config.toml` is read only by the Lasso split (g), the singular-value plot (b) and the bias-variance table (i). The other parts use the default `test_size=0.2` in `analysis.py`, so keep the config value at 0.2.
- The gradient-descent studies (e, f, h) fit the full data set, since they measure the distance to the closed-form solution on the same data.

---

## Results

`results/output_all_tasks.txt` contains all printed tables from a full run, including:

- the gradient checks,
- GD and optimizer convergence,
- the Lasso comparison with Scikit-Learn,
- the SGD timings,
- the final model selection.

The figures are in `figures/pngs/`. The file name starts with the method (`ols_`, `ridge_`, `lasso_`, `model_selection_`) and states what is plotted against what, e.g. `ridge_cv_mse_lambda_k_comparison.png`.

For high polynomial degrees and small $\lambda$, neither our gradient descent nor Scikit-Learn's coordinate descent converges for Lasso within the iteration budget, because the design matrix is badly conditioned. Lasso is therefore only cross-validated up to degree 10, and Scikit-Learn's convergence warnings are silenced.

---

## Declaration of use of generative AI

We used large language models (ChatGPT and Claude) as an aid during the project. They helped with debugging, with suggestions for code structure, with the plotting and table-printing code, and with writing this README. All suggestions were reviewed, tested and adapted by us before being included. We are responsible for all code, results and conclusions. The full declaration, following the course guidelines, is given in the report.

---

## Authors

Victor Berge Johansen
