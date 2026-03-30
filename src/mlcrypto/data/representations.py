from __future__ import annotations

import numpy as np


def _uint32_to_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint32)
    bytes_view = values.reshape(-1, 1).view(np.uint8)
    bits = np.unpackbits(bytes_view, axis=1, bitorder="big")
    return bits.astype(np.float32)


def _normalized_hamming_features(values: np.ndarray) -> np.ndarray:
    bits = _uint32_to_bits(values)
    total_weight = bits.sum(axis=1, keepdims=True) / 32.0
    left_weight = bits[:, :16].sum(axis=1, keepdims=True) / 16.0
    right_weight = bits[:, 16:].sum(axis=1, keepdims=True) / 16.0
    nibble_weights = bits.reshape(-1, 8, 4).sum(axis=2) / 4.0
    half_weight_gap = left_weight - right_weight
    return np.concatenate([total_weight, left_weight, right_weight, nibble_weights, half_weight_gap], axis=1).astype(np.float32)


def make_representation(name: str, p: np.ndarray, p_pair: np.ndarray, c: np.ndarray, c_pair: np.ndarray) -> np.ndarray:
    delta_c = np.bitwise_xor(c, c_pair).astype(np.uint32)

    if name == "delta":
        return _uint32_to_bits(delta_c)

    if name == "delta_stats":
        delta_bits = _uint32_to_bits(delta_c)
        stats = _normalized_hamming_features(delta_c)
        return np.concatenate([delta_bits, stats], axis=1)

    if name == "concat":
        return np.concatenate([_uint32_to_bits(c), _uint32_to_bits(c_pair)], axis=1)

    if name == "joint":
        return np.concatenate(
            [
                _uint32_to_bits(p),
                _uint32_to_bits(c),
                _uint32_to_bits(p_pair),
                _uint32_to_bits(c_pair),
            ],
            axis=1,
        )

    raise ValueError(f"Unknown representation: {name}")
