#!/usr/bin/env python3
"""Build a public-safe Tau-3 candidate identity from a final training receipt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_candidate_identity import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
