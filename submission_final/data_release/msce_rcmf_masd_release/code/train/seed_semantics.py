from __future__ import annotations

"""Explicit paper-facing semantics for the legacy repeat-id based seed scheme."""

from dataclasses import dataclass

from train.seeds import FULL_SEEDS, PILOT_SEEDS


PILOT_REPEAT_IDS = tuple(int(value) for value in PILOT_SEEDS)
FULL_REPEAT_IDS = tuple(int(value) for value in FULL_SEEDS)


@dataclass(frozen=True)
class LegacyRepeatProtocol:
    repeat_id: int

    def loader_seed(self, *, multiplier: int, offset: int) -> int:
        return int(self.repeat_id) * int(multiplier) + int(offset)

    def noise_seed(self, *, multiplier: int, offset: int) -> int:
        return int(self.repeat_id) * int(multiplier) + int(offset)


__all__ = [
    "FULL_REPEAT_IDS",
    "LegacyRepeatProtocol",
    "PILOT_REPEAT_IDS",
]
