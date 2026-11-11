 # torch_model.py
"""
Reference implementation of the classifier in PyTorch, for comparison with
our own network in experiments/classification.py. Same architecture, loss,
optimizer, epochs, batch size and regularization.
"""

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}


def build_torch_network(layers, activation_name):
    """Linear layers with the hidden activation between them, e.g. [784, 100, 10]."""
    modules = []
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        modules.append(nn.Linear(n_in, n_out))
        modules.append(ACTIVATIONS[activation_name]())
    return nn.Sequential(*modules[:-1])  # no activation after the last layer, see below


def penalty(net, reg_name, lmbda, n):
    """Same convention as regularizers.py: (lmbda/n) * sum over weights only, not biases."""
    weights = [p for name, p in net.named_parameters() if name.endswith("weight")]
    if reg_name == "l1":
        return lmbda / n * sum(W.abs().sum() for W in weights)
    return lmbda / n * sum((W ** 2).sum() for W in weights)


def train_torch_classifier(cfg, layers, activation_name, gamma, reg_name, lmbda, X, y):
    """Train with Adam and cross-entropy on mini-batches. Returns the network and loss per epoch."""
    c = cfg["classification"]
    torch.manual_seed(cfg["data"]["seed"])

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    n = len(y)

    net = build_torch_network(layers, activation_name)
    optimizer = torch.optim.Adam(net.parameters(), lr=gamma)
    loss_fn = nn.CrossEntropyLoss()  # Softmax + cross-entropy in one, takes raw outputs

    history = []
    for epoch in range(c["epochs"]):
        order = torch.randperm(n)                         # shuffle every epoch
        for start in range(0, n, c["batch_size"]):
            batch = order[start : start + c["batch_size"]]
            loss = loss_fn(net(X[batch]), y[batch])
            if lmbda > 0:
                loss = loss + penalty(net, reg_name, lmbda, n)
            optimizer.zero_grad()
            loss.backward()                               # automatic backpropagation
            optimizer.step()
        with torch.no_grad():
            history.append(loss_fn(net(X), y).item())     # loss without penalty, like network.fit
    return net, np.array(history)


def predict_labels_torch(net, X):
    """Predicted digit for each image."""
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()