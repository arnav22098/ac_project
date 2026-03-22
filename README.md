# ML-1 Course Project

Learning plaintext-ciphertext relations with machine learning using a reduced-round Speck32/64 cipher and a random-permutation baseline.

## What is included

- Reproducible paired-plaintext dataset generation
- Multiple input representations
- Three ML distinguishers: MLP, CNN, Siamese network
- Round-sweep experiment runner
- Plotting and report-ready summaries

## Quickstart

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Generate datasets:

```powershell
python scripts/generate_dataset.py --config configs/default.yaml
```

3. Run all experiments:

```powershell
python scripts/run_experiments.py --config configs/default.yaml
```

4. Build comparison plots:

```powershell
python scripts/make_plots.py --results-dir results/default
```

## Project layout

- `configs/`: experiment configuration
- `src/mlcrypto/crypto/`: cipher and random permutation baseline
- `src/mlcrypto/data/`: dataset generation and feature encodings
- `src/mlcrypto/models/`: neural distinguishers
- `src/mlcrypto/train/`: training, evaluation, experiment orchestration
- `scripts/`: command-line entrypoints
- `report/`: report outline and guidance

## Notes

- The random baseline is implemented as a lazy random permutation oracle over the queried plaintexts. This preserves permutation behavior on the sampled domain used during experiments.
- The default setup uses Speck32/64 because it is lightweight, fast to implement, and well-suited to round-wise cryptanalysis experiments.
