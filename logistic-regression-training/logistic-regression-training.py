import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)   # (N,)

    N, D = X.shape

    np.random.seed(1337)
    w = np.random.randn(D) * 0.1                 # (D,)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b                             # (N,)
        y_hat = _sigmoid(z)                       # (N,)

        dz = (y_hat - y)                          # (N,)
        dw = (X.T @ dz) / N                       # (D,)
        db = dz.mean()                            # scalar

        w -= lr * dw
        b -= lr * db

    return w, b