#!/usr/bin/env python3
"""Build private Tau-3 v3 scenario source JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_v3_scenarios import build_cli


if __name__ == "__main__":
    raise SystemExit(build_cli())
