from __future__ import annotations

import json
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


def _result_dir(config: dict, rounds: int, representation: str, model_name: str) -> Path:
    return Path(config["results"]["output_dir"]) / f"round_{rounds}" / representation / model_name


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
    output_dir = _result_dir(config, rounds, representation, model_name)
    if train_seed is not None:
        output_dir = output_dir / f"seed_{int(train_seed)}"
    artifacts = train_model(
        train_path=str(dataset_dir / "train.npz"),
        val_path=str(dataset_dir / "val.npz"),
        test_path=str(dataset_dir / "test.npz"),
        representation=representation,
        model_name=model_name,
        training_config=config["training"],
        output_dir=output_dir,
        device=config.get("device", "cpu"),
        train_seed=train_seed,
    )

    result = {
        "rounds": rounds,
        "representation": representation,
        "model": model_name,
        "train_seed": int(train_seed if train_seed is not None else config["seed"]),
        **artifacts.metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.history, handle, indent=2)
    return result


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
    with (results_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    frame = pd.DataFrame(records).sort_values(["model", "representation", "rounds", "train_seed"])
    frame.to_csv(results_dir / "summary_by_seed.csv", index=False)
    summary = _aggregate_summary(frame, config)
    summary.to_csv(results_dir / "summary.csv", index=False)
    build_plots(results_dir)
    print(f"Saved summary to {results_dir / 'summary.csv'}")


def build_plots(results_dir: str | Path) -> None:
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

    config_path = results_dir / "config_snapshot.json"
    evaluation = {}
    if config_path.exists():
        evaluation = json.loads(config_path.read_text(encoding="utf-8")).get("evaluation", {})
    balanced_threshold = float(evaluation.get("effective_balanced_accuracy", 0.60))
    auc_threshold = float(evaluation.get("effective_roc_auc", 0.60))
    minimum_seed_support = int(evaluation.get("minimum_effective_seed_count", 1))
    viable = frame[
        (frame["balanced_accuracy"] >= balanced_threshold)
        & (frame["roc_auc"] >= auc_threshold)
        & (frame["effective_seed_count"] >= minimum_seed_support)
    ].copy()
    if viable.empty:
        maximum_effective = pd.DataFrame(columns=["model", "representation", "max_effective_round"])
    else:
        maximum_effective = (
            viable.groupby(["model", "representation"], as_index=False)["rounds"]
            .max()
            .rename(columns={"rounds": "max_effective_round"})
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
        plt.bar(labels, maximum_effective["max_effective_round"])
        plt.ylabel("Max Effective Round")
        plt.title("Maximum Effective Round by Model/Representation")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        max_rounds_fig.savefig(results_dir / "max_effective_rounds.png", dpi=200)
        plt.close(max_rounds_fig)


def build_report_assets(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["results"]["output_dir"])
    data_dir = Path(config["data"]["output_dir"])
    build_plots(results_dir)

    report_dir = Path("report")
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    summary = summary.sort_values(["rounds", "representation", "model"]).copy()
    summary.to_csv(tables_dir / "summary_for_report.csv", index=False)
    if (results_dir / "summary_by_seed.csv").exists():
        pd.read_csv(results_dir / "summary_by_seed.csv").to_csv(tables_dir / "summary_by_seed_for_report.csv", index=False)

    best_by_round = summary.sort_values(["rounds", "roc_auc"], ascending=[True, False]).groupby("rounds").head(1).copy()
    best_by_round.to_csv(tables_dir / "best_by_round.csv", index=False)

    maximum_effective = pd.read_csv(results_dir / "max_effective_rounds.csv")
    maximum_effective.to_csv(tables_dir / "max_effective_rounds.csv", index=False)
    model_leaderboard = (
        summary.groupby("model", as_index=False)[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    model_leaderboard.to_csv(tables_dir / "model_leaderboard.csv", index=False)
    representation_leaderboard = (
        summary.groupby("representation", as_index=False)[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    representation_leaderboard.to_csv(tables_dir / "representation_leaderboard.csv", index=False)

    model_fig = plt.figure(figsize=(8, 5))
    plt.bar(model_leaderboard["model"], model_leaderboard["roc_auc"])
    plt.ylabel("Mean ROC-AUC Across All Rounds/Representations")
    plt.title("Overall Model Comparison")
    plt.ylim(0.45, 1.0)
    plt.tight_layout()
    model_fig.savefig(figures_dir / "model_leaderboard.png", dpi=200)
    plt.close(model_fig)

    representation_fig = plt.figure(figsize=(8, 5))
    plt.bar(representation_leaderboard["representation"], representation_leaderboard["roc_auc"])
    plt.ylabel("Mean ROC-AUC Across All Rounds/Models")
    plt.title("Overall Representation Comparison")
    plt.ylim(0.45, 1.0)
    plt.tight_layout()
    representation_fig.savefig(figures_dir / "representation_leaderboard.png", dpi=200)
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
    dataset_report = pd.DataFrame(rows).sort_values(["rounds", "split"])
    dataset_report.to_csv(tables_dir / "dataset_balance.csv", index=False)

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
    balance_fig.savefig(figures_dir / "dataset_balance.png", dpi=200)
    plt.close(balance_fig)

    for filename in [
        "accuracy_vs_rounds.png",
        "auc_vs_rounds.png",
        "balanced_accuracy_vs_rounds.png",
        "max_effective_rounds.png",
    ]:
        src = results_dir / filename
        if src.exists():
            dst = figures_dir / filename
            dst.write_bytes(src.read_bytes())
