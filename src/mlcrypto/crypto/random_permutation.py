from __future__ import annotations

from dataclasses import dataclass, field
import random


UINT32_MASK = 0xFFFFFFFF


@dataclass
class LazyRandomPermutation:
    seed: int
    mapping: dict[int, int] = field(default_factory=dict)
    used_outputs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def permute(self, value: int) -> int:
        value &= UINT32_MASK
        if value in self.mapping:
            return self.mapping[value]

        candidate = self.rng.getrandbits(32)
        while candidate in self.used_outputs:
            candidate = self.rng.getrandbits(32)

        self.mapping[value] = candidate
        self.used_outputs.add(candidate)
        return candidate
