from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mlcrypto.data.generation import load_bundle
from mlcrypto.data.representations import make_representation


class CryptoDataset(Dataset):
    def __init__(self, dataset_path: str, representation: str):
        bundle = load_bundle(dataset_path)
        features = make_representation(representation, bundle.p, bundle.p_pair, bundle.c, bundle.c_pair)
        labels = bundle.labels.astype(np.float32)

        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def infer_input_dim(dataset_path: str, representation: str) -> int:
    bundle = load_bundle(dataset_path)
    features = make_representation(representation, bundle.p, bundle.p_pair, bundle.c, bundle.c_pair)
    return int(features.shape[1])
