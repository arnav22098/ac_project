from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlcrypto.data.generation import generate_datasets_for_round
from mlcrypto.train.trainer import train_model
from mlcrypto.utils.config import load_config
from mlcrypto.utils.seed import set_seed


def generate_all_datasets(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    for rounds in config["data"]["rounds"]:
        generate_datasets_for_round(config, int(rounds))
        print(f"Generated datasets for round {rounds}")


def _training_seeds(config: dict) -> list[int]:
    evaluation = config.get("evaluation", {})
    if "train_seeds" in evaluation:
        return [int(seed) for seed in evaluation["train_seeds"]]
    return [int(config["seed"])]


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"rounds", "representation", "model", "train_seed"}
    return [column for column in frame.columns if column not in excluded]


def _aggregate_summary(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    metric_columns = _metric_columns(frame)
    mean_frame = frame.groupby(["rounds", "representation", "model"], as_index=False)[metric_columns].mean()
    std_frame = frame.groupby(["rounds", "representation", "model"], as_index=False)[metric_columns].std(ddof=0).fillna(0.0)
    merged = mean_frame.copy()
    for column in metric_columns:
        merged[f"{column}_std"] = std_frame[column]

    evaluation = config.get("evaluation", {})
    bal_threshold = float(evaluation.get("effective_balanced_accuracy", 0.60))
    auc_threshold = float(evaluation.get("effective_roc_auc", 0.60))
    effective_seed_counts = (
        frame.assign(
            seed_effective=lambda df: (
                (df["balanced_accuracy"] >= bal_threshold) & (df["roc_auc"] >= auc_threshold)
            ).astype(int)
        )
        .groupby(["rounds", "representation", "model"], as_index=False)["seed_effective"]
        .sum()
        .rename(columns={"seed_effective": "effective_seed_count"})
    )
    merged = merged.merge(effective_seed_counts, on=["rounds", "representation", "model"], how="left")
    merged["num_seeds"] = len(_training_seeds(config))
    return merged.sort_values(["model", "representation", "rounds"]).reset_index(drop=True)


def run_single_training(config_path: str, rounds: int, representation: str, model_name: str, train_seed: int | None = None) -> dict:
    config = load_config(config_path)
    set_seed(int(train_seed if train_seed is not None else config["seed"]))

    dataset_dir = Path(config["data"]["output_dir"]) / f"round_{rounds}"
    if not (dataset_dir / "train.npz").exists():
        generate_datasets_for_round(config, rounds)
    metrics = train_model(
        train_path=str(dataset_dir / "train.npz"),
        val_path=str(dataset_dir / "val.npz"),
        test_path=str(dataset_dir / "test.npz"),
        representation=representation,
        model_name=model_name,
        training_config=config["training"],
        device=config.get("device", "cpu"),
        train_seed=train_seed,
    )

    return {
        "rounds": rounds,
        "representation": representation,
        "model": model_name,
        "train_seed": int(train_seed if train_seed is not None else config["seed"]),
        **metrics,
    }


def run_all_experiments(config_path: str) -> None:
    config = load_config(config_path)
    generate_all_datasets(config_path)

    records = []
    train_seeds = _training_seeds(config)
    for rounds in config["data"]["rounds"]:
        for representation in config["representations"]:
            for model_name in config["models"]:
                for train_seed in train_seeds:
                    print(
                        f"Training rounds={rounds} representation={representation} "
                        f"model={model_name} seed={train_seed}"
                    )
                    result = run_single_training(config_path, int(rounds), representation, model_name, train_seed=train_seed)
                    records.append(result)

    results_dir = Path(config["results"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ["summary_by_seed.csv", "config_snapshot.json"]:
        stale_path = results_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_dir in results_dir.glob("round_*"):
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)
    frame = pd.DataFrame(records).sort_values(["model", "representation", "rounds", "train_seed"])
    summary = _aggregate_summary(frame, config)
    summary.to_csv(results_dir / "summary.csv", index=False)
    build_report_assets(config_path)
    print(f"Saved summary to {results_dir / 'summary.csv'}")


def build_plots(results_dir: str | Path, evaluation: dict | None = None) -> None:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    frame = pd.read_csv(summary_path)

    plt.style.use("ggplot")

    accuracy_fig = plt.figure(figsize=(10, 6))
    for (model_name, representation), subset in frame.groupby(["model", "representation"]):
        subset = subset.sort_values("rounds")
        plt.errorbar(
            subset["rounds"],
            subset["accuracy"],
            yerr=subset.get("accuracy_std", 0.0),
            marker="o",
            capsize=3,
            label=f"{model_name}-{representation}",
        )
    plt.xlabel("Rounds")
    plt.ylabel("Test Accuracy")
    plt.title("Distinguisher Accuracy vs Rounds")
    plt.legend()
    plt.tight_layout()
    accuracy_fig.savefig(results_dir / "accuracy_vs_rounds.png", dpi=200)
    plt.close(accuracy_fig)

    auc_fig = plt.figure(figsize=(10, 6))
    for (model_name, representation), subset in frame.groupby(["model", "representation"]):
        subset = subset.sort_values("rounds")
        plt.errorbar(
            subset["rounds"],
            subset["roc_auc"],
            yerr=subset.get("roc_auc_std", 0.0),
            marker="o",
            capsize=3,
            label=f"{model_name}-{representation}",
        )
    plt.xlabel("Rounds")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC vs Rounds")
    plt.legend()
    plt.tight_layout()
    auc_fig.savefig(results_dir / "auc_vs_rounds.png", dpi=200)
    plt.close(auc_fig)

    evaluation = evaluation or {}
    balanced_threshold = float(evaluation.get("effective_balanced_accuracy", 0.60))
    auc_threshold = float(evaluation.get("effective_roc_auc", 0.60))
    minimum_seed_support = int(evaluation.get("minimum_effective_seed_count", 1))
    viable = frame[
        (frame["balanced_accuracy"] >= balanced_threshold)
        & (frame["roc_auc"] >= auc_threshold)
        & (frame["effective_seed_count"] >= minimum_seed_support)
    ].copy()
    if viable.empty:
        maximum_effective = pd.DataFrame(columns=["model", "representation", "maximum_effective_round"])
    else:
        maximum_effective = (
            viable.groupby(["model", "representation"], as_index=False)["rounds"]
            .max()
            .rename(columns={"rounds": "maximum_effective_round"})
        )
    maximum_effective.to_csv(results_dir / "max_effective_rounds.csv", index=False)

    balanced_fig = plt.figure(figsize=(10, 6))
    for (model_name, representation), subset in frame.groupby(["model", "representation"]):
        subset = subset.sort_values("rounds")
        plt.errorbar(
            subset["rounds"],
            subset["balanced_accuracy"],
            yerr=subset.get("balanced_accuracy_std", 0.0),
            marker="o",
            capsize=3,
            label=f"{model_name}-{representation}",
        )
    plt.xlabel("Rounds")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy vs Rounds")
    plt.legend()
    plt.tight_layout()
    balanced_fig.savefig(results_dir / "balanced_accuracy_vs_rounds.png", dpi=200)
    plt.close(balanced_fig)

    if not maximum_effective.empty:
        max_rounds_fig = plt.figure(figsize=(10, 6))
        labels = [f"{row.model}-{row.representation}" for row in maximum_effective.itertuples(index=False)]
        plt.bar(labels, maximum_effective["maximum_effective_round"])
        plt.ylabel("Maximum Effective Round")
        plt.title("Maximum Effective Round by Model and Representation")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        max_rounds_fig.savefig(results_dir / "max_effective_rounds.png", dpi=200)
        plt.close(max_rounds_fig)


def build_report_assets(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["results"]["output_dir"])
    data_dir = Path(config["data"]["output_dir"])
    build_plots(results_dir, config.get("evaluation", {}))

    summary = pd.read_csv(results_dir / "summary.csv")
    summary = summary.sort_values(["rounds", "representation", "model"]).copy()

    best_by_round = (
        summary.sort_values(["rounds", "roc_auc"], ascending=[True, False])
        .groupby("rounds")
        .head(1)
        .loc[:, ["rounds", "model", "representation", "accuracy", "balanced_accuracy", "roc_auc"]]
        .rename(
            columns={
                "rounds": "round",
                "model": "best_model",
                "representation": "best_representation",
            }
        )
        .round({"accuracy": 4, "balanced_accuracy": 4, "roc_auc": 4})
    )
    best_by_round.to_csv(results_dir / "best_by_round.csv", index=False)

    model_ranking = (
        summary.groupby("model", as_index=False)[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
        .rename(
            columns={
                "accuracy": "average_accuracy",
                "balanced_accuracy": "average_balanced_accuracy",
                "roc_auc": "average_roc_auc",
            }
        )
        .round(4)
    )
    model_ranking.to_csv(results_dir / "model_ranking.csv", index=False)
    representation_ranking = (
        summary.groupby("representation", as_index=False)[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
        .rename(
            columns={
                "accuracy": "average_accuracy",
                "balanced_accuracy": "average_balanced_accuracy",
                "roc_auc": "average_roc_auc",
            }
        )
        .round(4)
    )
    representation_ranking.to_csv(results_dir / "representation_ranking.csv", index=False)

    model_fig = plt.figure(figsize=(8, 5))
    plt.bar(model_ranking["model"], model_ranking["average_roc_auc"])
    plt.ylabel("Average ROC-AUC")
    plt.title("Overall Model Ranking")
    plt.ylim(0.45, 1.0)
    plt.tight_layout()
    model_fig.savefig(results_dir / "model_ranking.png", dpi=200)
    plt.close(model_fig)

    representation_fig = plt.figure(figsize=(8, 5))
    plt.bar(representation_ranking["representation"], representation_ranking["average_roc_auc"])
    plt.ylabel("Average ROC-AUC")
    plt.title("Overall Representation Ranking")
    plt.ylim(0.45, 1.0)
    plt.tight_layout()
    representation_fig.savefig(results_dir / "representation_ranking.png", dpi=200)
    plt.close(representation_fig)

    rows = []
    for round_dir in sorted(data_dir.glob("round_*")):
        rounds = int(round_dir.name.split("_")[1])
        for split in ["train", "val", "test"]:
            bundle = np.load(round_dir / f"{split}.npz")
            labels = bundle["labels"]
            rows.append(
                {
                    "rounds": rounds,
                    "split": split,
                    "samples": int(labels.shape[0]),
                    "cipher_label_1": int(labels.sum()),
                    "random_label_0": int(labels.shape[0] - labels.sum()),
                    "positive_rate": float(labels.mean()),
                }
            )
    dataset_report = (
        pd.DataFrame(rows)
        .sort_values(["rounds", "split"])
        .rename(columns={"cipher_label_1": "cipher_samples", "random_label_0": "random_samples"})
        .round({"positive_rate": 4})
    )
    dataset_report.to_csv(results_dir / "dataset_balance.csv", index=False)

    balance_fig = plt.figure(figsize=(10, 6))
    split_order = ["train", "val", "test"]
    width = 0.25
    rounds = sorted(dataset_report["rounds"].unique())
    x = np.arange(len(rounds))
    for index, split in enumerate(split_order):
        subset = dataset_report[dataset_report["split"] == split].sort_values("rounds")
        offsets = x + (index - 1) * width
        plt.bar(offsets, subset["positive_rate"], width=width, label=split)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="ideal balance")
    plt.xticks(x, rounds)
    plt.ylim(0.45, 0.55)
    plt.xlabel("Rounds")
    plt.ylabel("Positive label rate")
    plt.title("Dataset Label Balance Across Splits")
    plt.legend()
    plt.tight_layout()
    balance_fig.savefig(results_dir / "dataset_balance.png", dpi=200)
    plt.close(balance_fig)
