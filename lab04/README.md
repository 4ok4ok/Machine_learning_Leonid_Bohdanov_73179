# ML Systems Engineering — Labs

![Python](https://img.shields.io/badge/python-3.11.8-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white)
![Reproducible](https://img.shields.io/badge/seed-42-success)
![Status](https://img.shields.io/badge/labs-5%2F11-blue)

Laboratory work for *Introduction to Machine Learning Systems* (Vijay Janapa Reddi). Each lab lives in its own folder with source code, results and a short report.

## Repository layout

```
repo/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                 # downloaded on demand, git-ignored
├── lab01/
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/ … lab11/
└── project/
    ├── README.md
    ├── src/
    ├── results/
    ├── data_card.md
    └── model_card.md
```

## Setup

```bash
git clone https://github.com/4ok4ok/Machine_learning_Leonid_Bohdanov_73179.git
cd Machine_learning_Leonid_Bohdanov_73179
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Lab 01

```bash
cd lab01/src
python print_versions.py     # -> results/versions.txt
python train_baselines.py    # -> results/baseline_accuracy.csv
python measure.py            # -> results/system_cost.csv, deployment_fit.csv, model*.joblib
```

## Labs

| Lab | Topic | Dataset | Report |
|---|---|---|---|
| 01 | Environment and first system measurements | Breast Cancer Wisconsin | [report](lab01/report.md) |
| 02 | — | Digits | [report](lab02/report.md) |
| 03 | — | Fashion-MNIST | — |
| 04 | Data pipeline and quality | UCI Adult | [report](lab04/report.md) |
| 05 | Frameworks and automatic differentiation | Two Moons | [report](lab05/report.md) |

## Reproducibility rules

- Seeds: `random`, `numpy`, `torch`, `torch.cuda` all set to `42`.
- Timings: one warm-up, then ≥ 5 repeats (100 for single-sample inference); the **median** is reported.
- Every lab writes its environment to `results/versions.txt`.
