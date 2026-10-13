import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def relu_backward(z):
    return (z > 0.0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, y_pred, eps=1e-15):
    # Cross-Entropy loss
    probs = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(probs), axis=1))


class TwoLayerMLP:
    def __init__(self, in_features=64, hidden_dim=16, num_classes=10, seed=42):
        np.random.seed(seed)
        self.hidden_dim = hidden_dim

        self.W1 = np.random.normal(0.0, 0.01, (in_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.normal(0.0, 0.01, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.normal(0.0, 0.01, (hidden_dim, num_classes))
        self.b3 = np.zeros(num_classes)

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def backward(self, y_true):
        N = y_true.shape[0]

        dZ3 = (self.A3 - y_true) / N
        self.dW3 = np.dot(self.A2.T, dZ3)
        self.db3 = np.sum(dZ3, axis=0)

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_backward(self.Z2)
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_backward(self.Z1)
        self.dW1 = np.dot(self.X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3