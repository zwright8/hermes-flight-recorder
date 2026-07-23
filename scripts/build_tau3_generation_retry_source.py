#!/usr/bin/env python3
"""Build a Tau-3 retry source JSONL from governed generation manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_generation_retry import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
