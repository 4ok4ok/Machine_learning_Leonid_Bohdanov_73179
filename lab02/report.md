# Lab 02: A Neural Network from Scratch

## 1. Goal
Design and implement a two-hidden-layer Multilayer Perceptron (MLP) from scratch using pure NumPy without deep learning frameworks. Verify the mathematical correctness of forward propagation, cross-entropy loss, and analytical gradients via numerical gradient checking. Train and evaluate the architecture across a grid of hidden layer widths (W in {16, 64}) and learning rates (eta in {0.001, 0.01, 0.1}) on the Digits dataset, and benchmark the execution latency of the forward vs. backward passes.

## 2. Method
- **Dataset:** Optical Recognition of Handwritten Digits (load_digits), normalized by dividing raw pixel values by 16.0 to map features to the [0, 1] interval. Labels are converted to 10-dimensional one-hot encoded vectors. Split 70/30 stratified with andom_state=42.
- **Architecture:** Input dimension 64 -> Linear -> ReLU(W) -> Linear -> ReLU(W) -> Linear -> Softmax(10). Weight tensors are initialized from N(0, 0.01^2) and bias vectors are initialized to zeros.
- **Verification Protocols:**
  - Output tensor shape checked to be (batch_size, 10), and each row verified to sum to 1.0 within 1e-6.
  - Untrained cross-entropy loss confirmed against the theoretical uniform guessing baseline ln(10) approx 2.3026.
  - Two-sided numerical gradient checking (eps = 1e-5) performed on 10 randomly selected weights (pass criteria: relative error < 1e-6).
- **Training Setup:** Mini-batch SGD with batch size 32, 50 epochs, and epoch-level data shuffling across 6 hyperparameter combinations.
- **Execution Profiling:** Forward and backward passes timed independently over 100 iterations following 1 warm-up run (reporting median latency).

## 3. Results

### Verification & Gradient Check Sanity
- **Untrained Loss:** Measured **2.3026**, matching theoretical -ln(0.1) approx 2.3026.
- **Numerical Gradient Check:** **10/10 passed** with relative error < 1e-8 (exceeding the required >= 9/10 threshold).

### Hyperparameter Grid Evaluation (50 Epochs)

| Hidden Width (W) | Learning Rate | Final Train Loss | Test Accuracy |
| :---: | :---: | :---: | :---: |
| 16 | 0.001 | 2.3001 | 10.19% |
| 16 | 0.010 | 0.8124 | 86.48% |
| 16 | 0.100 | 0.0841 | 95.19% |
| 64 | 0.001 | 2.2985 | 13.89% |
| 64 | 0.010 | 0.3211 | 93.33% |
| 64 | 0.100 | **0.0192** | **97.41%** |

![Training Loss Curves](results/loss_curves.png)

### Forward vs. Backward Latency Benchmark (Batch 32, Median of 100 runs)

| Pass | Median Latency (ms) | Relative Cost |
| :--- | :---: | :---: |
| **Forward Pass** | ~0.085 ms | 1.00x |
| **Backward Pass** | ~0.187 ms | **approx 2.20x** |

## 4. Three Conclusions in Your Own Words
1. **The backward pass computationally dominates due to dual GEMMs per layer:** The backward pass is roughly 2.2x slower than the forward pass. During the forward pass, each linear layer executes a single General Matrix Multiplication (X * W), whereas backpropagation requires two separate matrix multiplications per layer: one to compute weight gradients (A^T * delta) and another to propagate gradients back to previous layer activations (delta * W^T).
2. **Small weight initialization creates a dead zone for low learning rates:** Because weights are drawn from N(0, 0.01^2), pre-activations initially hover around zero where gradients are small. With learning rate 0.001, the network fails to break symmetry within 50 epochs and stays near random guess accuracy (~10-14%), while learning rate 0.1 quickly escapes this region and achieves over 97% test accuracy.
3. **Numerical gradient checking is essential for validating manual calculus:** Comparing analytical vector derivatives against centered finite differences (+-1e-5) confirmed relative error under 1e-8, proving that matrix transpose orientations and ReLU masking in backprop are strictly correct before running multi-epoch training.
