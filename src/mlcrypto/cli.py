import argparse

from mlcrypto.train.experiment import (
    build_plots,
    build_report_assets,
    generate_all_datasets,
    run_all_experiments,
    run_single_training,
)


def generate_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Generate Speck/random-permutation datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    generate_all_datasets(args.config)


def train_model_main() -> None:
    parser = argparse.ArgumentParser(description="Train one model/representation/round combo.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--rounds", required=True, type=int, help="Number of Speck rounds.")
    parser.add_argument("--representation", required=True, choices=["delta", "delta_stats", "concat", "joint"])
    parser.add_argument("--model", required=True, choices=["mlp", "cnn", "siamese"])
    args = parser.parse_args()
    run_single_training(args.config, args.rounds, args.representation, args.model)


def run_experiments_main() -> None:
    parser = argparse.ArgumentParser(description="Generate datasets and train all configured models.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_all_experiments(args.config)


def make_plots_main() -> None:
    parser = argparse.ArgumentParser(description="Create plots from experiment results.")
    parser.add_argument("--results-dir", required=True, help="Directory containing summary.csv")
    args = parser.parse_args()
    build_plots(args.results_dir)


def build_report_assets_main() -> None:
    parser = argparse.ArgumentParser(description="Create report figures and tables from config outputs.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    build_report_assets(args.config)
