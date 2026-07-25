#!/usr/bin/env python3
"""Run deterministic Tau-3 behavior probes against a local OpenAI-compatible endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.tau3_behavior_probes import main


if __name__ == "__main__":
    raise SystemExit(main())
