import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent              # папка lab02
RESULTS_DIR = BASE_DIR / "results"        # папка lab02/results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

def relu(z):
    return np.maximum(0.0, z)

def relu_backward(z):
    return (z > 0.0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, y_pred, eps=1e-15):
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

print("[1] Loading & preprocessing Digits dataset...")
X, y = load_digits(return_X_y=True)
X = X / 16.0

Y = np.zeros((y.size, 10))
Y[np.arange(y.size), y] = 1.0

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.3, stratify=y, random_state=42
)
print(f"    Train shape: {X_train.shape}, Test shape: {X_test.shape}")

print("\n[2] Checking forward pass sanity...")
model_check = TwoLayerMLP(hidden_dim=16, seed=42)
batch_X = X_train[:32]
batch_Y = Y_train[:32]
out = model_check.forward(batch_X)

assert out.shape == (32, 10), f"Wrong output shape: {out.shape}"
row_sums = np.sum(out, axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Rows do not sum to 1 within 1e-6!"
print(f"    Output shape: {out.shape} -> OK")
print(f"    Row sums check: min={row_sums.min():.6f}, max={row_sums.max():.6f} -> OK")

init_loss = compute_loss(batch_Y, out)
theo_loss = np.log(10)
print(f"\n[3] Untrained loss sanity check:")
print(f"    Measured loss:    {init_loss:.4f}")
print(f"    Theoretical ln(10): {theo_loss:.4f} -> OK")

print("\n[4] Running numerical gradient check (10 random weights, eps=1e-5)...")
net_gc = TwoLayerMLP(hidden_dim=16, seed=42)
X_mini = X_train[:8]
Y_mini = Y_train[:8]

net_gc.forward(X_mini)
net_gc.backward(Y_mini)

eps = 1e-5
passed_count = 0
weights_pool = [("W1", net_gc.W1, net_gc.dW1),
                ("W2", net_gc.W2, net_gc.dW2),
                ("W3", net_gc.W3, net_gc.dW3)]

for i in range(10):
    w_name, W, dW = weights_pool[np.random.randint(0, len(weights_pool))]
    idx = tuple(np.random.randint(0, s) for s in W.shape)

    orig_val = W[idx]

    W[idx] = orig_val + eps
    loss_plus = compute_loss(Y_mini, net_gc.forward(X_mini))

    W[idx] = orig_val - eps
    loss_minus = compute_loss(Y_mini, net_gc.forward(X_mini))

    W[idx] = orig_val

    num_grad = (loss_plus - loss_minus) / (2.0 * eps)
    ana_grad = dW[idx]

    rel_error = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-15)
    passed = rel_error < 1e-6
    if passed:
        passed_count += 1
    print(f"    Check {i+1:02d}: {w_name}{idx} | ana={ana_grad:.6e} num={num_grad:.6e} | rel_err={rel_error:.2e} -> {'PASS' if passed else 'FAIL'}")

print(f"    Gradient check result: {passed_count}/10 passed (requires >= 9/10)")

print("\n[5] Training models across parameter grid (50 epochs, batch=32)...")
widths = [16, 64]
lrs = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

plt.figure(figsize=(9, 5))
summary = []

for W in widths:
    for lr in lrs:
        tag = f"W={W}, lr={lr}"
        model = TwoLayerMLP(hidden_dim=W, seed=42)
        losses = []

        N = X_train.shape[0]
        for ep in range(epochs):
            indices = np.random.permutation(N)
            X_shuff = X_train[indices]
            Y_shuff = Y_train[indices]

            for i in range(0, N, batch_size):
                xb = X_shuff[i:i + batch_size]
                yb = Y_shuff[i:i + batch_size]
                model.forward(xb)
                model.backward(yb)
                model.step(lr)

            ep_loss = compute_loss(Y_train, model.forward(X_train))
            losses.append(ep_loss)

        test_pred = np.argmax(model.forward(X_test), axis=1)
        acc = np.mean(test_pred == y_test)

        plt.plot(range(1, epochs + 1), losses, label=f"{tag} (acc: {acc:.3f})")
        summary.append({
            "Hidden Width (W)": W,
            "Learning Rate": lr,
            "Final Train Loss": round(losses[-1], 4),
            "Test Accuracy": round(acc, 4)
        })
        print(f"    {tag:<16} | Final Loss: {losses[-1]:.4f} | Test Acc: {acc * 100:.2f}%")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curves (50 Epochs)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "loss_curves.png"
plt.savefig(plot_path, dpi=150)
print(f"\n    Saved loss curves to: {plot_path}")

df_results = pd.DataFrame(summary)
csv_path = RESULTS_DIR / "grid_results.csv"
df_results.to_csv(csv_path, index=False)
print(f"    Saved accuracies to:  {csv_path}")

print("\n[6] Timing Forward vs Backward passes (median of 100 runs)...")
timing_model = TwoLayerMLP(hidden_dim=64, seed=42)
bx, by = X_train[:32], Y_train[:32]

timing_model.forward(bx)
timing_model.backward(by)

fwd_times, bwd_times = [], []

for _ in range(100):
    t0 = time.perf_counter()
    timing_model.forward(bx)
    t1 = time.perf_counter()
    fwd_times.append((t1 - t0) * 1000)

    t0 = time.perf_counter()
    timing_model.backward(by)
    t1 = time.perf_counter()
    bwd_times.append((t1 - t0) * 1000)

med_fwd = np.median(fwd_times)
med_bwd = np.median(bwd_times)
ratio = med_bwd / med_fwd

print(f"    Forward pass median:  {med_fwd:.4f} ms")
print(f"    Backward pass median: {med_bwd:.4f} ms")
print(f"    Backward / Forward:   {ratio:.2f}x slower")

timing_path = RESULTS_DIR / "timing.txt"
with open(timing_path, "w") as f:
    f.write(f"Forward (ms): {med_fwd:.4f}\nBackward (ms): {med_bwd:.4f}\nRatio: {ratio:.2f}\n")
print(f"    Saved timing stats to: {timing_path}")

print("\nAll tasks completed successfully")