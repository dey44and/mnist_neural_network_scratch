import numpy as np


def get_predictions(A2):
    return np.argmax(A2, 0)


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def forward_prop(W1, b1, W2, b2, X):
    # Hidden layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    # Output layer
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
