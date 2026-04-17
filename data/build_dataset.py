from __future__ import annotations

"""Thin CLI wrapper for rebuilding the cleaned dataset and overlap report."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dedup import build_clean_dataset, write_outputs


def main() -> int:
    # Keep the public entry point minimal: the real construction logic lives in
    # data.dedup so the same pipeline can be reused by tests or notebooks.
    dataset, report, _ = build_clean_dataset()
    write_outputs(dataset, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
