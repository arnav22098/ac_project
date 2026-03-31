import argparse

from mlcrypto.train.experiment import (
    generate_all_datasets,
    run_all_experiments,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML cryptanalysis project runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-dataset", help="Generate Speck/random-permutation datasets.")
    generate_parser.add_argument("--config", required=True, help="Path to YAML config.")

    run_parser = subparsers.add_parser(
        "run-experiments",
        help="Generate datasets, train all configured models, and write final plots/tables.",
    )
    run_parser.add_argument("--config", required=True, help="Path to YAML config.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-dataset":
        generate_all_datasets(args.config)
        return

    if args.command == "run-experiments":
        run_all_experiments(args.config)
        return

    raise ValueError(f"Unknown command: {args.command}")
