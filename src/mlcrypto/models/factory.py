from __future__ import annotations

from mlcrypto.models.cnn import CNNDistinguisher
from mlcrypto.models.mlp import MLPDistinguisher
from mlcrypto.models.siamese import SiameseDistinguisher


def build_model(name: str, input_dim: int):
    if name == "mlp":
        return MLPDistinguisher(input_dim)
    if name == "cnn":
        return CNNDistinguisher(input_dim)
    if name == "siamese":
        return SiameseDistinguisher(input_dim)
    raise ValueError(f"Unknown model: {name}")
