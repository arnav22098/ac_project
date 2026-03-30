# Machine Learning Cryptanalysis of Reduced-Round Speck32/64

This repository contains a course project on learning plaintext-ciphertext relations with neural distinguishers. The experiments compare reduced-round `Speck32/64` against a random-permutation baseline and evaluate how far machine learning models remain effective as the number of encryption rounds increases.

## Highlights

- Reproducible paired-plaintext dataset generation pipeline
- Reproducible paired-plaintext dataset generation
- Multiple input representations, including a statistics-augmented difference encoding
- Three ML distinguishers: MLP, CNN, Siamese network
- Multi-seed round-sweep experiment runner
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

Build report assets:

```powershell
python scripts/build_report_assets.py --config configs/default.yaml
```

For a lightweight validation run, use:

```powershell
python scripts/run_experiments.py --config configs/fast_smoke.yaml
```

## Reproducibility Notes

- The random baseline is implemented as a lazy random permutation over the queried domain to preserve permutation behavior on sampled plaintexts.
- The default setup uses Speck32/64 because it is lightweight, fast to implement, and well-suited to round-wise cryptanalysis experiments.
- The default configuration aggregates results across two training seeds and reports mean/std metrics for stronger experimental rigor.

## Report Assets

- Final report: [`report/ML1_Report_Arnav_Batra.md`](report/ML1_Report_Arnav_Batra.md)
- Accuracy plot: [`report/figures/accuracy_vs_rounds.png`](report/figures/accuracy_vs_rounds.png)
- AUC plot: [`report/figures/auc_vs_rounds.png`](report/figures/auc_vs_rounds.png)
- Best-by-round summary: [`report/tables/best_by_round.csv`](report/tables/best_by_round.csv)

## Tech Stack

`Python`, `PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib`, `PyYAML`
