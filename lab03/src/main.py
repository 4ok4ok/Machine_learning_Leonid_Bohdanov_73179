import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
import thop

from models import FashionMLP, FashionCNN, FashionAttention

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = LAB_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n=== [1] Loading Fashion-MNIST ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_train, [55000, 5000], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

print("\n=== [2 & 3] Parameter and FLOPs Analysis ===")
models = {
    "MLP": FashionMLP().to(device),
    "CNN": FashionCNN().to(device),
    "Attention": FashionAttention().to(device)
}

hand_flops = {

    "MLP (Linear 784->256)": 2 * 784 * 256,
    "CNN (Conv2d 16->32)": 2 * 3 * 3 * 16 * 32 * 14 * 14,
    "Attention (QKV Proj)": 2 * 16 * 64 * (3 * 64)
}

print("Hand-calculated FLOPs for representative layers:")
for layer_name, flops in hand_flops.items():
    print(f"  {layer_name:<25}: {flops:,} FLOPs")

dummy_input = torch.randn(1, 1, 28, 28).to(device)
model_stats = []

for name, model in models.items():
    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    total_params = info.total_params

    model.eval()
    macs, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
    thop_flops = macs * 2

    model_stats.append({
        "Model": name,
        "Parameters": total_params,
        "thop_MACs": int(macs),
        "thop_FLOPs": int(thop_flops)
    })

df_stats = pd.DataFrame(model_stats)
print("\nModel Profile Summary:")
print(df_stats.to_string(index=False))
df_stats.to_csv(RESULTS_DIR / "model_complexity.csv", index=False)

print("\n=== Measuring Inference Latency ===")
latency_results = []

def measure_latency(model, input_tensor, runs):
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.median(times))

for name, model in models.items():
    dummy_b1 = torch.randn(1, 1, 28, 28).to(device)
    dummy_b128 = torch.randn(128, 1, 28, 28).to(device)

    lat_b1 = measure_latency(model, dummy_b1, runs=100)
    lat_b128 = measure_latency(model, dummy_b128, runs=20)

    print(f"  {name:<10} | Batch 1 (100 runs median): {lat_b1:.3f} ms | Batch 128 (20 runs median): {lat_b128:.3f} ms")
    latency_results.append({
        "Model": name,
        "Latency_Batch1_ms": round(lat_b1, 3),
        "Latency_Batch128_ms": round(lat_b128, 3)
    })

df_lat = pd.DataFrame(latency_results)
df_lat.to_csv(RESULTS_DIR / "latency_benchmark.csv", index=False)

print("\n=== [4] Training Models (5 epochs, Adam, lr=1e-3) ===")
epochs = 5
criterion = nn.CrossEntropyLoss()
training_results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    t_start = time.perf_counter()
    for ep in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        model.train()
        print(f"  Epoch {ep+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    total_train_time = time.perf_counter() - t_start

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"-> {name} Final Test Accuracy: {test_acc*100:.2f}% (Trained in {total_train_time:.1f}s)")

    training_results.append({
        "Model": name,
        "Test_Accuracy": round(test_acc * 100, 2),
        "Train_Time_s": round(total_train_time, 2)
    })

df_train = pd.DataFrame(training_results)
df_train.to_csv(RESULTS_DIR / "training_results.csv", index=False)

print("\n=== [5] Generating Trade-off Plots ===")
summary_df = df_stats.merge(df_lat, on="Model").merge(df_train, on="Model")
summary_df.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, row in summary_df.iterrows():
    axes[0].scatter(row["Parameters"] / 1000, row["Test_Accuracy"], s=140, label=row["Model"])
    axes[0].annotate(f"{row['Model']}\n({row['Test_Accuracy']}%)",
                     (row["Parameters"] / 1000, row["Test_Accuracy"]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
axes[0].set_title("Accuracy vs. Parameter Count")
axes[0].set_xlabel("Parameters (Thousands)")
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].grid(True, linestyle="--", alpha=0.5)

for i, row in summary_df.iterrows():
    axes[1].scatter(row["Latency_Batch128_ms"], row["Test_Accuracy"], s=140, label=row["Model"])
    axes[1].annotate(f"{row['Model']}\n({row['Latency_Batch128_ms']} ms)",
                     (row["Latency_Batch128_ms"], row["Test_Accuracy"]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
axes[1].set_title("Accuracy vs. Inference Latency (Batch 128)")
axes[1].set_xlabel("Latency (ms)")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "tradeoffs.png", dpi=150)
print(f"Saved trade-off plots to: {RESULTS_DIR / 'tradeoffs.png'}")
print("\n=== Done! All tasks completed. ===")