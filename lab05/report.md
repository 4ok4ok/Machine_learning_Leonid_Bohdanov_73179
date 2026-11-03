# Lab 05: Frameworks and automatic differentiation

Code: `src/lab5.py`. Raw numbers: `results/lab5_results.json`. Environment: `results/versions.txt`. Seed 42, CPU only.

## 1. Autograd engine (tasks 1-3)

`Value` is a scalar graph node holding `data`, `grad`, its parent nodes and a closure with the local derivative rule. Addition, multiplication, powers and ReLU (plus `log` and `sigmoid` for the BCE loss) create a new node while the forward pass runs, so the graph is built dynamically.

`backward()` sorts the graph topologically (iterative DFS), sets the output gradient to 1, visits nodes in reverse order and accumulates with `+=`. Accumulation matters when a node feeds several operations: for `u = t*t + t` at `t = 3` the engine gives `du/dt = 7`, which is `2t + 1`. Overwriting would give 3 or 6.

## 2. Gradient check (task 4)

Function: `((a*b + c)**2).relu().sum()` on 6-element random vectors, float64.

| Comparison | Max abs difference |
|---|---|
| engine vs numerical (central differences, eps 1e-6) | 1.05e-09 |
| engine vs PyTorch autograd | 0.0 |

Both are below the 1e-6 threshold.

## 3. Training on make_moons (task 5)

2 -> 8 -> 1, ReLU hidden layer, sigmoid output, mean BCE, full-batch SGD, lr 0.5, 50 epochs, 100 samples. Both implementations start from the same weights and use float64.

| | Median of 5 runs | Final loss | Accuracy |
|---|---|---|---|
| Custom engine | 7.021 s | 0.2802 | 0.89 |
| PyTorch | 0.0461 s | 0.2802 | 0.89 |

The loss curves agree to 3.3e-16 per epoch, so the engine computes the same thing as PyTorch. PyTorch is about 152x faster. The custom engine allocates a Python object per scalar operation and walks every node in Python, while PyTorch runs a few batched matrix operations in compiled kernels.

## 4. Eager vs compiled, graph inspection (task 6)

**Graph.** The autograd graph of `Linear(16,32) -> ReLU -> Linear(32,1) -> sum` is `SumBackward0 -> AddmmBackward0 -> ReluBackward0 -> AddmmBackward0`, with `AccumulateGrad` leaves for weights and biases and `TBackward0` nodes for the transposed weights. `torch.jit.trace` records the module calls as a `prim::CallMethod` sequence.

**Tensors.** Inputs and parameters are float32 and contiguous (for example the input has shape (64, 16) and strides (16, 1)). `input.t()` is a view with strides (1, 16), so it is not contiguous and no copy was made.

**Eager vs compiled.** `torch.compile` failed on this machine with an `InductorError` (the Inductor backend needs a working C++ compiler toolchain), so I used `torch.jit.trace` as the task allows. 20 warm-up calls, then the median of 5 timing blocks of 200 calls.

| Case | Eager | jit.trace | Speed-up |
|---|---|---|---|
| MLP 256-512-512-10, batch 256 | 2840.6 us | 2567.6 us | 1.11x |
| Pointwise chain (sin, cos, exp, sigmoid, tanh), 1024x1024 | 9191.4 us | 10225.8 us | 0.90x |

**Non-contiguous view** (2048x2048 float32):

| Operation | Time |
|---|---|
| `a + 1` | 2.56 ms |
| `a.t() + 1` | 2.56 ms |
| `a.t().contiguous() + 1` | 18.06 ms |
| `a[:, ::2] + 1` | 1.74 ms |
| `a.sum(1)` | 0.96 ms |
| `a.t().sum(1)` | 2.20 ms |
| `a @ a` | 173.45 ms |
| `a.t() @ a` | 179.44 ms |

Elementwise addition on a transposed view costs the same as on a contiguous tensor, because the kernel iterates using strides. The reduction over the transposed view is about 2.3x slower, since it reads memory with a stride of one row. Matrix multiplication is unaffected, as BLAS handles transposed operands directly. Forcing a copy with `.contiguous()` is the expensive option here (about 7x slower than the plain operation), so it only pays off when many operations follow on the same data.

## 5. When does compilation help?

My measurements do not show a clear benefit from tracing. The MLP gains 11% and the pointwise chain gets 10% slower, which is within what I would expect from run-to-run noise on a CPU. `jit.trace` records the operator sequence but does not fuse operators by itself, so the traced graph executes the same kernels as eager. The MLP is dominated by large matrix multiplications that already run in optimised library code, so there is little left to remove.

What I expect from a real compiler (`torch.compile` with Inductor, which I could not run), based on the framework chapter rather than on my own measurements: it helps most when a model has many small elementwise operations that can be fused into one kernel, which avoids writing and reading intermediate tensors and saves per-operation framework overhead; it also helps in long-running training or serving loops with fixed input shapes. It helps little for models dominated by large matmuls, for short scripts, or when shapes change often, because compilation is a one-time cost and shape changes trigger recompilation.

Limitation: the compile experiment is incomplete, since `torch.compile` could not be run here. To test it, install a C++ toolchain (MSVC Build Tools on Windows) or run the script on Linux/Colab.
