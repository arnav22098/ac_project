from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlcrypto.crypto.random_permutation import LazyRandomPermutation
from mlcrypto.crypto.speck import Speck32_64


UINT32_MASK = 0xFFFFFFFF


@dataclass
class DatasetBundle:
    p: np.ndarray
    p_pair: np.ndarray
    c: np.ndarray
    c_pair: np.ndarray
    labels: np.ndarray


def _sample_uint32(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, UINT32_MASK + 1, size=size, dtype=np.uint32)


def _build_fixed_key(config: dict) -> tuple[int, int, int, int]:
    key_words = config["data"]["key_schedule"]["key"]
    if len(key_words) != 4:
        raise ValueError("Speck32/64 requires exactly four 16-bit key words.")
    return tuple(int(word) & 0xFFFF for word in key_words)


def generate_split(
    size: int,
    delta_p: int,
    rounds: int,
    key_words: tuple[int, int, int, int],
    seed: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    plaintexts = _sample_uint32(rng, size)
    paired_plaintexts = np.bitwise_xor(plaintexts, np.uint32(delta_p & UINT32_MASK))
    labels = rng.integers(0, 2, size=size, dtype=np.uint8)

    cipher = Speck32_64(rounds=rounds, key_words=key_words)
    permutation = LazyRandomPermutation(seed=seed ^ (rounds << 8) ^ 0xA5A5A5A5)

    c = np.empty(size, dtype=np.uint32)
    c_pair = np.empty(size, dtype=np.uint32)

    for index in range(size):
        p = int(plaintexts[index])
        p2 = int(paired_plaintexts[index])
        if labels[index] == 1:
            c[index] = cipher.encrypt(p)
            c_pair[index] = cipher.encrypt(p2)
        else:
            c[index] = permutation.permute(p)
            c_pair[index] = permutation.permute(p2)

    return DatasetBundle(
        p=plaintexts,
        p_pair=paired_plaintexts.astype(np.uint32),
        c=c,
        c_pair=c_pair,
        labels=labels,
    )


def save_bundle(bundle: DatasetBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        p=bundle.p,
        p_pair=bundle.p_pair,
        c=bundle.c,
        c_pair=bundle.c_pair,
        labels=bundle.labels,
    )


def load_bundle(dataset_path: Path) -> DatasetBundle:
    data = np.load(dataset_path)
    return DatasetBundle(
        p=data["p"],
        p_pair=data["p_pair"],
        c=data["c"],
        c_pair=data["c_pair"],
        labels=data["labels"],
    )


def generate_datasets_for_round(config: dict, rounds: int) -> dict[str, Path]:
    seed = int(config["seed"])
    delta_p = int(config["data"]["delta_p"], 16) if isinstance(config["data"]["delta_p"], str) else int(config["data"]["delta_p"])
    output_dir = Path(config["data"]["output_dir"]) / f"round_{rounds}"
    key_words = _build_fixed_key(config)

    splits = {
        "train": int(config["data"]["train_size"]),
        "val": int(config["data"]["val_size"]),
        "test": int(config["data"]["test_size"]),
    }
    split_seed_offsets = {"train": 0, "val": 1000, "test": 2000}
    generated_paths: dict[str, Path] = {}

    for split_name, split_size in splits.items():
        bundle = generate_split(
            size=split_size,
            delta_p=delta_p,
            rounds=rounds,
            key_words=key_words,
            seed=seed + split_seed_offsets[split_name] + rounds,
        )
        path = output_dir / f"{split_name}.npz"
        save_bundle(bundle, path)
        generated_paths[split_name] = path

    return generated_paths
