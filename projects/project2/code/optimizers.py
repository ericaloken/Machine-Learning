# optimizers.py
"""
Gradient-based optimizers for linear models and neural networks.

All optimizers work on any nested structure of arrays (pytrees), so the same class can update
a signle coefficient vector (``theta`` for linear regression) or
a list of weight matrices and bias vectors (``W`` and ``b`` for a neural network).

Typical usage:
------
optimizer = Adam(gamma=0.001) # initializing also resets the internal state
optimizer.reset()             # to avoid carrying over state from a previous run
for X_batch, y_batch in batches:
    grads = compute_gradients(X_batch, y_batch)
    params = optimizer.step(params, grads)




Explaination of pytrees and why we use them:
A nested structure of arrays, such as a list of weight matrices and bias vectors for a NN is called a pytree.
We use pytrees to allow the optimizers to work on any nested structure of arrays,
so the same class can update a single coefficient vector (theta for linear regression) or
a list of weight matrices and bias vectors (W and b for a neural network).
The alternative would be to write separate optimizers for each type of model or
boolean flag to indicate type of model or similar approaches.

We use the function tree_map to apply a function to each leaf of the pytree (each array in the nested structure).
For instance:

    def update_param(param, grad):
        return param - gamma_t * grad

    tree_map(update_param, params, grads)
will apply the function update_param to each pair of corresponding leaves in the pytrees params and grads
"""

import numpy as np
from jax.tree_util import tree_map
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class for gradient-based optimizers.

    Holds the hyperparameters and internal state shared by all optimizers.
    Subclasses implement :meth:`_update`, which applies one update with a
    given learning rate. :meth:`step` handles the step counter, the
    learning-rate schedule and lazy initialization of the state.

    Parameters
    ----------
    gamma : float
        Learning rate. Used as the constant learning rate when `schedule`
        is None.
    beta1 : float, optional
        Decay rate for the first moment (Adam) or the momentum coefficient
        (Momentum). Default is 0.9.
    beta2 : float, optional
        Decay rate for the second moment (RMSprop, Adam). Default is 0.999.
        RMSprop is often run with beta2 = 0.9.
    epsilon : float, optional
        Small constant added to denominators to avoid division by zero.
        Default is 1e-8.
    a, b : float, optional
        Parameters of the inverse time-decay schedule, see `schedule`.
        Default is 0.0.
    schedule : {None, "inverse_time"} or callable, optional
        Learning-rate schedule as a function of the step number t, starting
        at t = 1.

        - None: constant learning rate gamma_t = gamma.
        - "inverse_time": gamma_t = a / (t + b).
        - callable: gamma_t = schedule(t).

        Default is None.

    Attributes
    ----------
    t : int
        Number of steps taken since the last reset.
    m : pytree of jax.Array or None
        First-moment estimate (Adam) or velocity (Momentum), same structure
        as the parameters. None until the first step.
    v : pytree of jax.Array or None
        Second-moment estimate or accumulated squared gradients (AdaGrad,
        RMSprop, Adam), same structure as the parameters. None until the
        first step.
    """

    def __init__(
        self,
        gamma,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        a=0.0,
        b=0.0,
        schedule=None,
    ):
        # raise value errors here

        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.schedule = schedule

        self.reset()

    def initialize(self, params):
        """
        Initialize the optimizer state to zeros matching `params`.

        Called automatically on the first :meth:`step` after a reset.

        Parameters
        ----------
        params : pytree of array_like
            Model parameters, e.g. ``theta`` or a list of ``(W, b)`` tuples.
        """
        self.m = tree_map(np.zeros_like, params)
        self.v = tree_map(np.zeros_like, params)

    def reset(self):
        """
        Clear the step counter and the optimizer state.

        Call this before each new training run, so that state from a
        previous run (e.g. in a hyperparameter search) is not carried over.
        """
        self.t = 0
        self.m = None
        self.v = None

    def learning_rate(self):
        """
        Learning rate for the current step.

        Returns
        -------
        float
            The learning rate `gamma_t` for the current step.
        """
        if self.schedule is None:
            return self.gamma
        if self.schedule == "inverse_time":
            return self.a / (self.t + self.b)
        return self.schedule(self.t)

    def step(self, params, grads):
        """
        Take one optimization step.


        Parameters
        ----------
        params : pytree of array_like
            Current parameters.
        grads : pytree of array_like
            Gradients of the cost with respect to `params`, same structure.

        Returns
        -------
        pytree of jax.Array
            Updated parameters, same structure as `params`.
        """
        self.t += 1
        if self.m is None or self.v is None:
            self.initialize(params)
        gamma_t = self.learning_rate()
        return self._update(params, grads, gamma_t)

    @abstractmethod
    def _update(self, params, grads, gamma_t):
        """
        Apply one update with learning rate `gamma_t`. Implemented by subclasses.

        Parameters
        ----------
        params : pytree of array_like
            Current parameters.
        grads : pytree of array_like
            Gradients, same structure as `params`.
        gamma_t : float
            Learning rate for this step.

        Returns
        -------
        pytree of jax.Array
            Updated parameters.
        """


class GradientDescent(Optimizer):
    """
    Plain gradient descent (GD).

    Update rule: p_{t+1} = p_t - gamma_t * grad_t

    Gives stochastic gradient descent when used with mini-batches.
    """

    def _update(self, params, grads, gamma_t):
        def update_param(param, grad):
            # GD step for a single parameter array
            return param - gamma_t * grad

        # tree_map applies update_param to each leaf of the pytree (each)
        return tree_map(update_param, params, grads)


class Momentum(Optimizer):
    """
    GD with momentum.

    Update rule: m_{t+1} = beta1 * m_t + (1 - beta1) * grad_t
                 p_{t+1} = p_t - gamma_t * m_{t+1

    ``beta1`` is the momentum coefficient. m is often called the velocity.
    """

    def _update(self, params, grads, gamma_t):
        step_size = gamma_t / (
            1 - self.beta1
        )  # rescale step size to match classical momentum

        def update_m(m, g):
            # running average of gradients (momentum)
            return self.beta1 * m + (1 - self.beta1) * g

        def update_param(param, m):
            # step in the direction of the running average
            return param - step_size * m

        self.m = tree_map(update_m, self.m, grads)
        return tree_map(update_param, params, self.m)


class AdaGrad(Optimizer):
    """
    Adaptive gradient (AdaGrad).

    Update rule: v_{t+1} = v_t + grad_t^2
                 p_{t+1} = p_t - gamma_t * grad_t / (sqrt(v_{t+1}) + epsilon)

    Each parameter gets its own learning rate, which shrinks as squared
    gradients accumulate in v.
    """

    def _update(self, params, grads, gamma_t):
        def update_v(v, g):
            # accumulate squared gradients
            return v + g**2

        def update_param(param, grad, v):
            # step with adaptive learning rate
            return param - gamma_t * grad / (np.sqrt(v) + self.epsilon)

        self.v = tree_map(update_v, self.v, grads)
        return tree_map(update_param, params, grads, self.v)


class RMSprop(Optimizer):
    """
    Root mean square propagation (RMSprop).

    Update rule: v_{t+1} = beta2 * v_t + (1 - beta2) * grad_t^2
                 p_{t+1} = p_t - gamma_t * grad_t / (sqrt(v_{t+1}) + epsilon)

    Like AdaGrad, but v is a running average, so old gradients are forgotten
    and the learning rate does not shrink towards zero.
    """

    def _update(self, params, grads, gamma_t):
        def update_v(v, grad):
            # running average of the squared gradient
            return self.beta2 * v + (1 - self.beta2) * grad**2

        def update_param(p, grad, v):
            # step scaled by the recent gradient size
            return p - gamma_t * grad / (np.sqrt(v) + self.epsilon)

        self.v = tree_map(update_v, self.v, grads)
        return tree_map(update_param, params, grads, self.v)


class Adam(Optimizer):
    """
    Adaptive moment estimation (Adam).
    Update rule: m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
                v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2
                m_hat = m_t / (1 - beta1^t)
                v_hat = v_t / (1 - beta2^t)
                p_{t+1} = p_t - gamma_t * m_hat / (sqrt(v_hat) + epsilon)
    The bias corrections are baked into the step size and epsilon
    for speed.

    Combines momentum (m) with RMSprop scaling (v). The bias corrections
    m_hat and v_hat compensate for m and v starting at zero.
    """

    def _update(self, params, grads, gamma_t):
        beta1, beta2 = self.beta1, self.beta2
        bc1 = 1 - beta1**self.t
        bc2 = 1 - beta2**self.t
        alpha = gamma_t * np.sqrt(bc2) / bc1
        eps = self.epsilon * np.sqrt(bc2)

        def update_m(m, grad):
            m *= beta1
            m += (1 - beta1) * grad
            return m

        def update_v(v, grad):
            v *= beta2
            v += (1 - beta2) * np.square(grad)
            return v

        def update_param(p, m, v):
            d = np.sqrt(v)
            d += eps
            np.divide(m, d, out=d)
            d *= alpha
            return p - d  # new array, caller's params untouched

        self.m = tree_map(update_m, self.m, grads)
        self.v = tree_map(update_v, self.v, grads)
        return tree_map(update_param, params, self.m, self.v)


OPTIMIZERS = {
    "gd": GradientDescent,
    "sgd": GradientDescent,
    "momentum": Momentum,
    "adagrad": AdaGrad,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def make_optimizer(name, gamma, **kwargs):
    """
    Creates an optimizer instance based on name
    """
    try:
        optimizer_class = OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown optimizer name: {name}. Available optimizers: {list(OPTIMIZERS.keys())}"
        )
    return optimizer_class(gamma, **kwargs)
