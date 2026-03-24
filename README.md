# Machine Learning Cryptanalysis of Reduced-Round Speck32/64

This repository contains a course project on learning plaintext-ciphertext relations with neural distinguishers. The experiments compare reduced-round `Speck32/64` against a random-permutation baseline and evaluate how far machine learning models remain effective as the number of encryption rounds increases.

## Highlights

- Reproducible paired-plaintext dataset generation pipeline
- Multiple input representations: `concat`, `delta`, and `joint`
- Three learned distinguishers: `MLP`, `CNN`, and `Siamese`
- Round-sweep experiment runner for reduced-round analysis
- Report assets, plots, and summary tables ready for submission

## Key Results

- Best overall setting: `Siamese + delta representation` at 3 rounds
- Peak test accuracy: `95.0%`
- Peak ROC-AUC: `0.986`
- Strong performance remains at 4 rounds, with balanced accuracy near `0.79`
- Effective distinguishers begin to degrade around 5 rounds and above, which matches the expected increase in cipher hardness

The generated report assets and summary tables live in [`report/`](report/) and include comparison plots across rounds, representations, and model families.

## Repository Structure

- `configs/`: experiment configurations for full and smoke runs
- `scripts/`: dataset generation, training, evaluation, plotting, and report asset builders
- `src/mlcrypto/crypto/`: reduced-round Speck and random-permutation baseline
- `src/mlcrypto/models/`: neural model definitions
- `src/mlcrypto/train/`: experiment orchestration, metrics, and training loops
- `src/mlcrypto/utils/`: config loading and reproducibility helpers
- `report/`: report markdown, PDF export, figures, and summary tables

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate a dataset:

```powershell
python scripts/generate_dataset.py --config configs/default.yaml
```

Run the experiment sweep:

```powershell
python scripts/run_experiments.py --config configs/default.yaml
```

Build plots from the generated results:

```powershell
python scripts/make_plots.py --results-dir results/default
```

For a lightweight validation run, use:

```powershell
python scripts/run_experiments.py --config configs/fast_smoke.yaml
```

## Reproducibility Notes

- The random baseline is implemented as a lazy random permutation over the queried domain to preserve permutation behavior on sampled plaintexts.
- The project uses `Speck32/64` because it is lightweight to simulate and well suited for round-wise cryptanalysis experiments.
- Seeds and configuration values are centralized so the same setup can be rerun consistently.

## Report Assets

- Final report: [`report/ML1_Report_Arnav_Batra.pdf`](report/ML1_Report_Arnav_Batra.pdf)
- Accuracy plot: [`report/figures/accuracy_vs_rounds.png`](report/figures/accuracy_vs_rounds.png)
- AUC plot: [`report/figures/auc_vs_rounds.png`](report/figures/auc_vs_rounds.png)
- Best-by-round summary: [`report/tables/best_by_round.csv`](report/tables/best_by_round.csv)

## Tech Stack

`Python`, `PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib`, `PyYAML`
