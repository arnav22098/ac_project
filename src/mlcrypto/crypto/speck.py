from __future__ import annotations

from dataclasses import dataclass


WORD_MASK = 0xFFFF


def _rotl16(value: int, shift: int) -> int:
    return ((value << shift) & WORD_MASK) | (value >> (16 - shift))


def _rotr16(value: int, shift: int) -> int:
    return (value >> shift) | ((value << (16 - shift)) & WORD_MASK)


@dataclass(frozen=True)
class Speck32_64:
    rounds: int
    key_words: tuple[int, int, int, int]

    def _encrypt_round(self, x: int, y: int, key: int) -> tuple[int, int]:
        x = ((_rotr16(x, 7) + y) & WORD_MASK) ^ key
        y = _rotl16(y, 2) ^ x
        return x, y

    def _expand_key(self) -> list[int]:
        round_keys = [self.key_words[-1]]
        l_schedule = [self.key_words[i] for i in reversed(range(len(self.key_words) - 1))]
        for i in range(self.rounds - 1):
            new_l, new_k = self._encrypt_round(l_schedule[i], round_keys[i], i)
            l_schedule.append(new_l)
            round_keys.append(new_k)
        return round_keys

    def encrypt(self, plaintext: int) -> int:
        x = (plaintext >> 16) & WORD_MASK
        y = plaintext & WORD_MASK
        round_keys = self._expand_key()
        for key in round_keys:
            x, y = self._encrypt_round(x, y, key)
        return ((x & WORD_MASK) << 16) | (y & WORD_MASK)
