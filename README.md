# Machine Learning Cryptanalysis of Reduced-Round Speck32/64

This repository contains a course project on learning plaintext-ciphertext relations with neural distinguishers. The experiments compare reduced-round `Speck32/64` against a random-permutation baseline and evaluate how far machine learning models remain effective as the number of encryption rounds increases.

## Highlights

- Reproducible paired-plaintext dataset generation
- Multiple input representations, including a statistics-augmented difference encoding
- Three ML distinguishers: MLP, CNN, Siamese network
- Multi-seed round-sweep experiment runner
- Final plots, summary tables, and report files ready for submission

## Key Results

- Best overall setting: `Siamese + delta representation` at 3 rounds
- Peak test accuracy: `95.0%`
- Peak ROC-AUC: `0.986`
- Strong performance remains at 4 rounds, with balanced accuracy near `0.79`
- Effective distinguishers begin to degrade around 5 rounds and above, which matches the expected increase in cipher hardness

The final experimental results and plots live in [`results/default/`](results/default/), and the final written report lives in [`report/`](report/).

## Repository Structure

- `configs/`: experiment configuration
- `results/default/`: final experimental results and plots for submission
- `main.py`: single command-line entrypoint for the whole project
- `src/mlcrypto/data/`: dataset generation and feature representations
- `src/mlcrypto/crypto/`: reduced-round Speck and random-permutation baseline
- `src/mlcrypto/models/`: neural model definitions
- `src/mlcrypto/train/`: experiment orchestration, metrics, and training loops
- `src/mlcrypto/utils/`: config loading and reproducibility helpers
- `report/`: final report files

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate a dataset:

```powershell
python main.py generate-dataset --config configs/default.yaml
```

Run the experiment sweep:

```powershell
python main.py run-experiments --config configs/default.yaml
```
This single command also writes the final CSV summaries and plots used in the report.

## Reproducibility Notes

- The random baseline is implemented as a lazy random permutation over the queried domain to preserve permutation behavior on sampled plaintexts.
- The default setup uses Speck32/64 because it is lightweight, fast to implement, and well-suited to round-wise cryptanalysis experiments.
- The default configuration aggregates results across two training seeds and reports mean/std metrics for stronger experimental rigor.

## Report Assets

- Final report: [`report/2022098_2022592_FinalReport.md`](report/2022098_2022592_FinalReport.md)
- Final PDF: [`report/2022098_2022592_FinalReport.pdf`](report/2022098_2022592_FinalReport.pdf)
- Results summary: [`results/default/summary.csv`](results/default/summary.csv)
- Accuracy plot: [`results/default/accuracy_vs_rounds.png`](results/default/accuracy_vs_rounds.png)
- AUC plot: [`results/default/auc_vs_rounds.png`](results/default/auc_vs_rounds.png)
- Best-by-round summary: [`results/default/best_by_round.csv`](results/default/best_by_round.csv)

## Tech Stack

`Python`, `PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib`, `PyYAML`
