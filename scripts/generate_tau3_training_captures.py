#!/usr/bin/env python3
"""CLI wrapper for deterministic Tau-3 training capture generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_capture_generation import _main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_main())
