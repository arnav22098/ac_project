from __future__ import annotations

import torch
from torch import nn


class SiameseDistinguisher(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("Siamese model expects an even input dimension.")
        branch_dim = input_dim // 2
        self.branch_dim = branch_dim
        self.encoder = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = x[:, : self.branch_dim]
        right = x[:, self.branch_dim :]
        left_embed = self.encoder(left)
        right_embed = self.encoder(right)
        diff = torch.abs(left_embed - right_embed)
        joined = torch.cat([left_embed, right_embed, diff], dim=1)
        return self.head(joined)
