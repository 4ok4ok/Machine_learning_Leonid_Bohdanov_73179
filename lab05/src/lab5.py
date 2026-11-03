import json
import math
import platform
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
RESULTS = {}

_here = Path(__file__).resolve().parent
OUT = (_here.parent if _here.name == "src" else _here) / "results"


class Value:
    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sigmoid(self):
        z = max(-500.0, min(500.0, self.data))
        out = Value(1.0 / (1.0 + math.exp(-z)), (self,), "sigmoid")

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    def backward(self):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in seen:
                    stack.append((p, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def vsum(values):
    total = Value(0.0)
    for v in values:
        total = total + v
    return total


# task 4

def f_np(a, b, c):
    return float(np.maximum((a * b + c) ** 2, 0.0).sum())


def num_grad(f, args, idx, eps=1e-6):
    g = np.zeros_like(args[idx])
    for i in range(g.size):
        hi = [x.copy() for x in args]
        lo = [x.copy() for x in args]
        hi[idx].flat[i] += eps
        lo[idx].flat[i] -= eps
        g.flat[i] = (f(*hi) - f(*lo)) / (2 * eps)
    return g


def task4():
    rng = np.random.default_rng(SEED)
    a_np, b_np, c_np = (rng.normal(size=6) for _ in range(3))
    args = [a_np, b_np, c_np]

    a, b, c = ([Value(x) for x in arr] for arr in args)
    out = vsum([((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c)])
    out.backward()
    mine = [np.array([v.grad for v in vs]) for vs in (a, b, c)]

    num = [num_grad(f_np, args, i) for i in range(3)]
    d_num = max(np.abs(m - n).max() for m, n in zip(mine, num))
    print(f"[4] max abs diff vs numerical: {d_num:.3e}")
    assert d_num < 1e-6
    res = {"diff_vs_numerical": float(d_num)}

    try:
        import torch
        ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in args]
        ((ts[0] * ts[1] + ts[2]) ** 2).relu().sum().backward()
        d_t = max(np.abs(m - t.grad.numpy()).max() for m, t in zip(mine, ts))
        print(f"[4] max abs diff vs torch autograd: {d_t:.3e}")
        assert d_t < 1e-6
        res["diff_vs_torch"] = float(d_t)
    except ImportError:
        print("[4] torch not installed, skipping autograd comparison")
    RESULTS["task4"] = res


# task 5

N, EPOCHS, LR, HIDDEN = 100, 50, 0.5, 8


def init_weights(seed=SEED):
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, 0.5, (HIDDEN, 2)), b1=np.zeros(HIDDEN),
                W2=rng.normal(0, 0.5, (1, HIDDEN)), b2=np.zeros(1))


def train_mine(X, y, w):
    W1 = [[Value(v) for v in row] for row in w["W1"]]
    b1 = [Value(v) for v in w["b1"]]
    W2 = [Value(v) for v in w["W2"][0]]
    b2 = Value(w["b2"][0])
    params = [p for row in W1 for p in row] + b1 + W2 + [b2]
    losses = []
    for _ in range(EPOCHS):
        terms = []
        for xi, yi in zip(X, y):
            h = [(vsum(W1[j][i] * float(xi[i]) for i in range(2)) + b1[j]).relu()
                 for j in range(HIDDEN)]
            p = (vsum(W2[j] * h[j] for j in range(HIDDEN)) + b2).sigmoid()
            terms.append(-(yi * (p + 1e-12).log() + (1 - yi) * ((1 - p) + 1e-12).log()))
        loss = vsum(terms) * (1.0 / len(X))
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= LR * p.grad
        losses.append(loss.data)

    correct = 0
    for xi, yi in zip(X, y):
        h = [max(0.0, sum(W1[j][i].data * xi[i] for i in range(2)) + b1[j].data)
             for j in range(HIDDEN)]
        z = sum(W2[j].data * h[j] for j in range(HIDDEN)) + b2.data
        correct += int((z > 0) == bool(yi))
    return losses, correct / len(X)


def train_torch(X, y, w):
    import torch
    Xt, yt = torch.tensor(X), torch.tensor(y)
    W1, b1, W2, b2 = (torch.tensor(w[k], requires_grad=True) for k in ("W1", "b1", "W2", "b2"))
    params = [W1, b1, W2, b2]
    losses = []
    for _ in range(EPOCHS):
        h = torch.relu(Xt @ W1.T + b1)
        p = torch.sigmoid(h @ W2.T + b2).squeeze(1)
        loss = -(yt * torch.log(p + 1e-12) + (1 - yt) * torch.log(1 - p + 1e-12)).mean()
        for q in params:
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in params:
                q -= LR * q.grad
        losses.append(loss.item())
    with torch.no_grad():
        z = (torch.relu(Xt @ W1.T + b1) @ W2.T + b2).squeeze(1)
        acc = ((z > 0).double() == yt).double().mean().item()
    return losses, acc


def timed(fn, runs=5):
    times, last = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, last


def task5():
    X, y = make_moons(n_samples=N, noise=0.1, random_state=SEED)
    w = init_weights()
    med_m, runs_m, (loss_m, acc_m) = timed(lambda: train_mine(X, y, w))
    print(f"[5] custom engine: median {med_m:.3f}s, runs {[round(t, 3) for t in runs_m]}")
    print(f"    loss {loss_m[0]:.4f} -> {loss_m[-1]:.4f}, acc {acc_m:.2f}")
    res = {"mine_median_s": med_m, "mine_loss_first": loss_m[0],
           "mine_loss_last": loss_m[-1], "mine_acc": acc_m}
    try:
        import torch  # noqa: F401
        med_t, runs_t, (loss_t, acc_t) = timed(lambda: train_torch(X, y, w))
        diff = max(abs(a - b) for a, b in zip(loss_m, loss_t))
        print(f"[5] pytorch: median {med_t:.4f}s, runs {[round(t, 4) for t in runs_t]}")
        print(f"    loss {loss_t[0]:.4f} -> {loss_t[-1]:.4f}, acc {acc_t:.2f}")
        print(f"    max loss diff per epoch: {diff:.2e}, pytorch ~{med_m / med_t:.0f}x faster")
        res.update(torch_median_s=med_t, torch_loss_last=loss_t[-1], torch_acc=acc_t,
                   loss_diff=diff, speedup=med_m / med_t)
    except ImportError:
        print("[5] torch not installed, skipping")
    RESULTS["task5"] = res


# task 6

def bench(fn, x, warmup=20, iters=200, repeats=5):
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        meds = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            meds.append((time.perf_counter() - t0) / iters)
    return statistics.median(meds)


def task6():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[6] torch not installed, skipping")
        return
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"torch": torch.__version__, "device": dev}

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    x = torch.randn(64, 16, device=dev)
    y = model(x).sum()

    print("\n[6a] autograd graph:")
    seen = set()

    def walk(fn, depth=0):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        print("  " * (depth + 1) + type(fn).__name__)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)
    walk(y.grad_fn)

    print("\n[6a] traced graph:")
    print(torch.jit.trace(model, x).graph)

    print("[6a] tensor info:")
    tensors = {"input": x, "input.t()": x.t()}
    tensors.update(dict(model.named_parameters()))
    res["tensors"] = {}
    for name, t in tensors.items():
        print(f"  {name:12s} shape={tuple(t.shape)} dtype={t.dtype} "
              f"contiguous={t.is_contiguous()} stride={t.stride()}")
        res["tensors"][name] = [list(t.shape), str(t.dtype), t.is_contiguous()]

    big = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512),
                        nn.ReLU(), nn.Linear(512, 10)).to(dev).eval()

    def pointwise(t):
        return torch.tanh(torch.sin(t) * torch.cos(t) + t.exp().sigmoid()) * 0.5 + t * t

    cases = {"mlp": (big, torch.randn(256, 256, device=dev)),
             "pointwise": (pointwise, torch.randn(1024, 1024, device=dev))}
    res["eager_vs_compiled"] = {}
    print("\n[6b] eager vs compiled:")
    for name, (fn, inp) in cases.items():
        t_eager = bench(fn, inp)
        mode = "torch.compile"
        try:
            t0 = time.perf_counter()
            comp = torch.compile(fn)
            with torch.no_grad():
                comp(inp)
            compile_s = time.perf_counter() - t0
        except Exception as e:
            print(f"  torch.compile failed ({type(e).__name__}), using jit.trace")
            mode = "jit.trace"
            t0 = time.perf_counter()
            with torch.no_grad():
                comp = torch.jit.trace(fn, inp)
            compile_s = time.perf_counter() - t0
        t_comp = bench(comp, inp)
        print(f"  {name:10s} eager {t_eager * 1e6:9.1f} us | {mode} {t_comp * 1e6:9.1f} us "
              f"| x{t_eager / t_comp:.2f} | compile time {compile_s:.2f}s")
        res["eager_vs_compiled"][name] = dict(
            eager_us=t_eager * 1e6, compiled_us=t_comp * 1e6, mode=mode,
            speedup=t_eager / t_comp, compile_s=compile_s)

    print("\n[6c] contiguous vs non-contiguous:")
    a = torch.randn(2048, 2048, device=dev)
    at, asl = a.t(), a[:, ::2]

    def t_op(fn, reps=20):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if a.is_cuda:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    tests = [("a + 1", lambda: a + 1),
             ("a.t() + 1", lambda: at + 1),
             ("a.t().contiguous() + 1", lambda: at.contiguous() + 1),
             ("a[:, ::2] + 1", lambda: asl + 1),
             ("a.sum(1)", lambda: a.sum(1)),
             ("a.t().sum(1)", lambda: at.sum(1)),
             ("a @ a", lambda: a @ a),
             ("a.t() @ a", lambda: at @ a)]
    res["contiguity_ms"] = {}
    for name, fn in tests:
        t = t_op(fn) * 1e3
        print(f"  {name:24s} {t:9.2f} ms")
        res["contiguity_ms"][name] = t
    RESULTS["task6"] = res


if __name__ == "__main__":
    x, y = Value(2.0), Value(3.0)
    z = (x * y + x ** 2).relu()
    z.backward()
    assert (x.grad, y.grad) == (7.0, 2.0)
    t = Value(3.0)
    u = t * t + t
    u.backward()
    assert t.grad == 7.0

    task4()
    task5()
    task6()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "lab5_results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    import sklearn
    lines = [f"python {platform.python_version()}", f"numpy {np.__version__}",
             f"scikit-learn {sklearn.__version__}", f"platform {platform.platform()}"]
    try:
        import torch
        lines.append(f"torch {torch.__version__}")
    except ImportError:
        pass
    (OUT / "versions.txt").write_text("\n".join(lines) + "\n")