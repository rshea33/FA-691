import numpy as np


def gbm(S0, mu, sigma, T, N=5):
    "Returns a single path of a geometric Brownian motion."
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W) * np.sqrt(dt) 
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)
    return S
